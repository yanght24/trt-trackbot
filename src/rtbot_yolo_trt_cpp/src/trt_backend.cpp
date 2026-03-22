// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#include "rtbot_yolo_trt_cpp/trt_backend.hpp"
#include "rtbot_yolo_trt_cpp/cuda_preprocess.hpp"

#include <NvInferPlugin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace rtbot_yolo_trt
{

// ── 辅助函数 ────────────────────────────────────────────────────────

static size_t volume(const nvinfer1::Dims & d)
{
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
  return v;
}

static size_t typeSize(nvinfer1::DataType dt)
{
  switch (dt) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT8:  return 1;
    default: return 4;
  }
}

static std::unique_ptr<void, TrtBackend::CudaDeleter> cudaAlloc(size_t bytes)
{
  void * ptr = nullptr;
  if (cudaMalloc(&ptr, bytes) != cudaSuccess)
    throw std::runtime_error("cudaMalloc failed: " + std::to_string(bytes) + " bytes");
  return std::unique_ptr<void, TrtBackend::CudaDeleter>(ptr);
}

static std::unique_ptr<void, TrtBackend::PinnedDeleter> pinnedAlloc(size_t bytes)
{
  void * ptr = nullptr;
  if (cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault) != cudaSuccess)
    throw std::runtime_error("cudaHostAlloc failed: " + std::to_string(bytes) + " bytes");
  return std::unique_ptr<void, TrtBackend::PinnedDeleter>(ptr);
}

// ── 构造函数 ────────────────────────────────────────────────────────

TrtBackend::TrtBackend(const std::string & engine_path)
{
  // 初始化 NvInfer 插件（EfficientNMS_TRT 需要）
  initLibNvInferPlugins(&logger_, "");

  // 读取 engine 文件
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) throw std::runtime_error("Cannot open: " + engine_path);
  const auto sz = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(sz);
  if (!file.read(data.data(), sz)) throw std::runtime_error("Read failed: " + engine_path);

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) throw std::runtime_error("createInferRuntime failed");

  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");

  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("createExecutionContext failed");

  if (cudaStreamCreate(&stream_) != cudaSuccess)
    throw std::runtime_error("cudaStreamCreate failed");

  discoverTensors();
  allocateBuffers();

  std::cerr << "[TrtBackend] End2end engine loaded: " << engine_path
            << "  input=" << input_h_ << "x" << input_w_
            << "  max_det=" << max_det_ << std::endl;
}

TrtBackend::~TrtBackend()
{
  if (stream_) { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); }
}

// ── 张量自动发现（TRT 10 API）────────────────────────────────────

void TrtBackend::discoverTensors()
{
  const int nb = engine_->getNbIOTensors();
  std::cerr << "[TrtBackend] IO tensors (" << nb << "):" << std::endl;

  for (int i = 0; i < nb; ++i) {
    const char * name = engine_->getIOTensorName(i);
    auto shape = engine_->getTensorShape(name);
    auto mode  = engine_->getTensorIOMode(name);
    auto dtype = engine_->getTensorDataType(name);

    std::cerr << "  " << name
              << " mode=" << (mode == nvinfer1::TensorIOMode::kINPUT ? "IN" : "OUT")
              << " shape=[";
    for (int d = 0; d < shape.nbDims; ++d) std::cerr << (d ? "," : "") << shape.d[d];
    std::cerr << "] dtype=" << static_cast<int>(dtype) << std::endl;

    const std::string sname(name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      // 输入张量: "images" 或 "image_arrays" [1, 3, H, W]
      name_input_ = sname;
      input_h_ = shape.d[2];
      input_w_ = shape.d[3];
      input_size_bytes_ = volume(shape) * typeSize(dtype);
    }
    else if (sname == "num" || sname == "num_dets") {
      name_num_ = sname;
      num_bytes_ = volume(shape) * typeSize(dtype);
    }
    else if (sname == "boxes" || sname == "det_boxes") {
      name_boxes_ = sname;
      max_det_ = shape.d[1];   // shape: [1, max_det, 4]
      boxes_bytes_ = volume(shape) * typeSize(dtype);
    }
    else if (sname == "scores" || sname == "det_scores") {
      name_scores_ = sname;
      scores_bytes_ = volume(shape) * typeSize(dtype);
    }
    else if (sname == "classes" || sname == "det_classes") {
      name_classes_ = sname;
      classes_bytes_ = volume(shape) * typeSize(dtype);
    }
  }

  // 验证所有张量名称已找到
  if (name_input_.empty())  throw std::runtime_error("[TrtBackend] Input tensor not found");
  if (name_num_.empty())    throw std::runtime_error("[TrtBackend] 'num' tensor not found — is this an end2end engine?");
  if (name_boxes_.empty())  throw std::runtime_error("[TrtBackend] 'boxes' tensor not found");
  if (name_scores_.empty()) throw std::runtime_error("[TrtBackend] 'scores' tensor not found");
  if (name_classes_.empty())throw std::runtime_error("[TrtBackend] 'classes' tensor not found");
}

// ── 缓冲区分配 ──────────────────────────────────────────────────

void TrtBackend::allocateBuffers()
{
  // Device
  d_input_   = cudaAlloc(input_size_bytes_);
  d_num_     = cudaAlloc(num_bytes_);
  d_boxes_   = cudaAlloc(boxes_bytes_);
  d_scores_  = cudaAlloc(scores_bytes_);
  d_classes_ = cudaAlloc(classes_bytes_);

  // Pinned host（D2H 异步传输用）
  h_num_pinned_     = pinnedAlloc(num_bytes_);
  h_boxes_pinned_   = pinnedAlloc(boxes_bytes_);
  h_scores_pinned_  = pinnedAlloc(scores_bytes_);
  h_classes_pinned_ = pinnedAlloc(classes_bytes_);
}

// ── 推理 ────────────────────────────────────────────────────────

TrtBackend::InferResult TrtBackend::infer(
  const uint8_t * h_bgr, int src_w, int src_h, float conf_threshold)
{
  const size_t src_bytes = static_cast<size_t>(src_w) * src_h * 3;

  // 按需扩展源图缓冲区
  if (src_bytes > src_pinned_capacity_) {
    h_src_pinned_ = pinnedAlloc(src_bytes);
    src_pinned_capacity_ = src_bytes;
  }
  if (src_bytes > d_src_capacity_) {
    d_src_img_ = cudaAlloc(src_bytes);
    d_src_capacity_ = src_bytes;
  }

  // 拷贝原始 BGR 到 pinned 内存，然后异步 H2D
  std::memcpy(h_src_pinned_.get(), h_bgr, src_bytes);
  cudaMemcpyAsync(d_src_img_.get(), h_src_pinned_.get(), src_bytes,
                  cudaMemcpyHostToDevice, stream_);

  // 计算 letterbox 参数
  const float r_w = static_cast<float>(input_w_) / src_w;
  const float r_h = static_cast<float>(input_h_) / src_h;
  const float scale = std::min(r_w, r_h);
  const float pad_x = (input_w_ - src_w * scale) / 2.0f;
  const float pad_y = (input_h_ - src_h * scale) / 2.0f;

  // GPU letterbox + BGR→RGB + 归一化 + HWC→CHW
  cudaPreprocessLetterbox(
    static_cast<const uint8_t *>(d_src_img_.get()), src_w, src_h,
    static_cast<float *>(d_input_.get()), input_w_, input_h_,
    pad_x, pad_y, scale, stream_);

  // 设置张量地址（TRT 10 API）
  context_->setTensorAddress(name_input_.c_str(),   d_input_.get());
  context_->setTensorAddress(name_num_.c_str(),     d_num_.get());
  context_->setTensorAddress(name_boxes_.c_str(),   d_boxes_.get());
  context_->setTensorAddress(name_scores_.c_str(),  d_scores_.get());
  context_->setTensorAddress(name_classes_.c_str(), d_classes_.get());

  // 入队推理
  if (!context_->enqueueV3(stream_)) {
    std::cerr << "[TrtBackend] enqueueV3 failed" << std::endl;
    return {};
  }

  // D2H: 所有输出异步拷贝到 pinned 内存
  cudaMemcpyAsync(h_num_pinned_.get(),     d_num_.get(),     num_bytes_,     cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(h_boxes_pinned_.get(),   d_boxes_.get(),   boxes_bytes_,   cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(h_scores_pinned_.get(),  d_scores_.get(),  scores_bytes_,  cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(h_classes_pinned_.get(), d_classes_.get(), classes_bytes_, cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // 解析 end2end 输出
  const auto * h_num     = static_cast<const int32_t *>(h_num_pinned_.get());
  const auto * h_boxes   = static_cast<const float *>(h_boxes_pinned_.get());
  const auto * h_scores  = static_cast<const float *>(h_scores_pinned_.get());
  const auto * h_classes = static_cast<const int32_t *>(h_classes_pinned_.get());

  const int n_det = std::min(static_cast<int>(h_num[0]), max_det_);

  // 将框从 letterbox 空间映射到原图坐标
  std::vector<Detection> dets;
  dets.reserve(n_det);

  for (int i = 0; i < n_det; ++i) {
    // end2end 引擎保证 h_num[0] 个框都过了内部 conf 阈值，这里做二次过滤
    if (h_scores[i] < conf_threshold) [[likely]] continue;

    // end2end 输出的框是 letterbox 像素空间中的 x1,y1,x2,y2
    float bx1 = (h_boxes[i * 4 + 0] - pad_x) / scale;
    float by1 = (h_boxes[i * 4 + 1] - pad_y) / scale;
    float bx2 = (h_boxes[i * 4 + 2] - pad_x) / scale;
    float by2 = (h_boxes[i * 4 + 3] - pad_y) / scale;

    // 裁剪到原图范围
    bx1 = std::clamp(bx1, 0.0f, static_cast<float>(src_w));
    by1 = std::clamp(by1, 0.0f, static_cast<float>(src_h));
    bx2 = std::clamp(bx2, 0.0f, static_cast<float>(src_w));
    by2 = std::clamp(by2, 0.0f, static_cast<float>(src_h));

    // 跳过退化框
    if ((bx2 - bx1) < 1.0f || (by2 - by1) < 1.0f) continue;

    // C++20 designated initializer — 比逐字段赋值更清晰
    dets.push_back({
      .x1 = bx1, .y1 = by1, .x2 = bx2, .y2 = by2,
      .score = h_scores[i],
      .class_id = static_cast<int>(h_classes[i])
    });
  }

  return {
    std::move(dets),
    PreprocessInfo{
      .orig_w = src_w, .orig_h = src_h,
      .scale = scale, .pad_x = pad_x, .pad_y = pad_y
    }
  };
}

}  // namespace rtbot_yolo_trt
