// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#include "rtbot_yolo_trt_cpp/trt_engine.hpp"
#include "rtbot_yolo_trt_cpp/cuda_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace rtbot_yolo_trt
{

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

static std::unique_ptr<void, TrtEngine::CudaDeleter> cudaAlloc(size_t bytes)
{
  void * ptr = nullptr;
  if (cudaMalloc(&ptr, bytes) != cudaSuccess)
    throw std::runtime_error("cudaMalloc failed: " + std::to_string(bytes) + " bytes");
  return std::unique_ptr<void, TrtEngine::CudaDeleter>(ptr);
}

TrtEngine::TrtEngine(const std::string & engine_path)
{
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

  // 自动发现 IO 张量（TRT 10 API）
  const int nb = engine_->getNbIOTensors();
  std::cerr << "[TrtEngine] IO tensors (" << nb << "):" << std::endl;
  for (int i = 0; i < nb; ++i) {
    const char * name = engine_->getIOTensorName(i);
    auto shape = engine_->getTensorShape(name);
    auto mode  = engine_->getTensorIOMode(name);
    auto dtype = engine_->getTensorDataType(name);

    std::cerr << "  " << name << " mode=" << (mode == nvinfer1::TensorIOMode::kINPUT ? "IN" : "OUT")
              << " shape=[";
    for (int d = 0; d < shape.nbDims; ++d) std::cerr << (d ? "," : "") << shape.d[d];
    std::cerr << "]" << std::endl;

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      name_input_ = name;
      input_c_ = shape.d[1];
      input_h_ = shape.d[2];
      input_w_ = shape.d[3];
      input_size_bytes_ = volume(shape) * typeSize(dtype);
    } else {
      name_output_ = name;
      // 标准 YOLO: [1, 84, num_anchors]
      const int dim1 = shape.d[1];  // 84
      num_anchors_    = shape.d[2]; // 例如 19320
      num_classes_    = dim1 - 4;   // 80
      output_size_bytes_ = volume(shape) * typeSize(dtype);
    }
  }

  if (name_input_.empty() || name_output_.empty())
    throw std::runtime_error("Could not find input/output tensors");

  allocateBuffers();

  std::cerr << "[TrtEngine] Loaded: " << engine_path
            << "  input=" << input_c_ << "x" << input_h_ << "x" << input_w_
            << "  num_classes=" << num_classes_
            << "  num_anchors=" << num_anchors_ << std::endl;
}

TrtEngine::~TrtEngine()
{
  if (stream_) { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); }
}

void TrtEngine::allocateBuffers()
{
  d_input_  = cudaAlloc(input_size_bytes_);
  d_output_ = cudaAlloc(output_size_bytes_);

  // Pinned host 输入缓冲区
  void * pinned_in = nullptr;
  if (cudaHostAlloc(&pinned_in, input_size_bytes_, cudaHostAllocDefault) != cudaSuccess)
    throw std::runtime_error("cudaHostAlloc failed for pinned input");
  h_input_pinned_.reset(pinned_in);

  // Pinned host 输出缓冲区（支持真正的异步 D2H DMA）
  void * pinned_out = nullptr;
  if (cudaHostAlloc(&pinned_out, output_size_bytes_, cudaHostAllocDefault) != cudaSuccess)
    throw std::runtime_error("cudaHostAlloc failed for pinned output");
  h_output_pinned_.reset(pinned_out);
  h_output_ptr_ = static_cast<float *>(pinned_out);
  output_num_floats_ = output_size_bytes_ / sizeof(float);
}

std::vector<Detection> TrtEngine::infer(const float * input_chw, float conf_threshold)
{
  // H2D（如果调用者用了 pinned_input()，input_chw == pinned_input()）
  // Pinned 内存支持真正的异步 DMA，无需中间拷贝。
  const void * src = (input_chw == pinned_input()) ? h_input_pinned_.get() : input_chw;
  cudaMemcpyAsync(d_input_.get(), src, input_size_bytes_,
                  cudaMemcpyHostToDevice, stream_);

  // 设置张量地址并执行推理（TRT 10 API）
  context_->setTensorAddress(name_input_.c_str(),  d_input_.get());
  context_->setTensorAddress(name_output_.c_str(), d_output_.get());

  if (!context_->enqueueV3(stream_)) {
    std::cerr << "[TrtEngine] enqueueV3 failed" << std::endl;
    return {};
  }

  // D2H
  cudaMemcpyAsync(h_output_ptr_, d_output_.get(), output_size_bytes_,
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // 解析 raw 输出 [1, 84, num_anchors]
  const float * out = h_output_ptr_;
  const int nc = num_classes_;
  const int na = num_anchors_;

  std::vector<Detection> dets;
  dets.reserve(512);

  // 按行指针访问以优化缓存局部性
  const float * row_cx = out + 0 * na;
  const float * row_cy = out + 1 * na;
  const float * row_w  = out + 2 * na;
  const float * row_h  = out + 3 * na;

  for (int j = 0; j < na; ++j) {
    // 找最高类别得分——扫描该 anchor 的所有类别行
    float max_score = 0.0f;
    int max_class = 0;
    for (int c = 0; c < nc; ++c) {
      const float s = out[(4 + c) * na + j];
      if (s > max_score) { max_score = s; max_class = c; }
    }

    // 早期拒绝：使用实际的 conf_threshold（而非 0.01）
    // 95%+ 的 anchor 会在此被丢弃，大幅减少后续 NMS 工作量
    if (max_score < conf_threshold) [[likely]] continue;  // 绝大多数 anchor 分数很低

    const float cx = row_cx[j];
    const float cy = row_cy[j];
    const float w  = row_w[j];
    const float h  = row_h[j];

    dets.push_back({
      .x1 = cx - w * 0.5f, .y1 = cy - h * 0.5f,
      .x2 = cx + w * 0.5f, .y2 = cy + h * 0.5f,
      .score = max_score, .class_id = max_class
    });
  }

  return dets;
}

std::vector<Detection> TrtEngine::inferWithGpuPreprocess(
  const uint8_t * h_bgr, int src_w, int src_h, float conf_threshold)
{
  const size_t src_bytes = static_cast<size_t>(src_w) * src_h * 3;

  // 按需扩展 pinned host 和 device 源缓冲区
  if (src_bytes > src_pinned_capacity_) {
    void * pinned = nullptr;
    if (cudaHostAlloc(&pinned, src_bytes, cudaHostAllocDefault) != cudaSuccess)
      throw std::runtime_error("cudaHostAlloc failed for source image");
    h_src_pinned_.reset(pinned);
    src_pinned_capacity_ = src_bytes;
  }
  if (src_bytes > d_src_capacity_) {
    d_src_img_ = cudaAlloc(src_bytes);
    d_src_capacity_ = src_bytes;
  }

  // 拷贝原始 BGR 到 pinned，然后异步 H2D
  std::memcpy(h_src_pinned_.get(), h_bgr, src_bytes);
  cudaMemcpyAsync(d_src_img_.get(), h_src_pinned_.get(), src_bytes,
                  cudaMemcpyHostToDevice, stream_);

  // GPU letterbox + BGR→RGB + 归一化 + HWC→CHW
  const float r_w = static_cast<float>(input_w_) / src_w;
  const float r_h = static_cast<float>(input_h_) / src_h;
  const float scale = std::min(r_w, r_h);
  const float pad_x = (input_w_ - src_w * scale) / 2.0f;
  const float pad_y = (input_h_ - src_h * scale) / 2.0f;

  cudaPreprocessLetterbox(
    static_cast<const uint8_t *>(d_src_img_.get()), src_w, src_h,
    static_cast<float *>(d_input_.get()), input_w_, input_h_,
    pad_x, pad_y, scale, stream_);

  // TRT 推理（输入已在 device 上）
  context_->setTensorAddress(name_input_.c_str(),  d_input_.get());
  context_->setTensorAddress(name_output_.c_str(), d_output_.get());

  if (!context_->enqueueV3(stream_)) {
    std::cerr << "[TrtEngine] enqueueV3 failed" << std::endl;
    return {};
  }

  // D2H 输出到 pinned 内存
  cudaMemcpyAsync(h_output_ptr_, d_output_.get(), output_size_bytes_,
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // 解析输出（与 infer() 相同逻辑）
  const float * out = h_output_ptr_;
  const int nc = num_classes_;
  const int na = num_anchors_;
  const float * row_cx = out + 0 * na;
  const float * row_cy = out + 1 * na;
  const float * row_w  = out + 2 * na;
  const float * row_h  = out + 3 * na;

  std::vector<Detection> dets;
  dets.reserve(512);
  for (int j = 0; j < na; ++j) {
    float max_score = 0.0f;
    int max_class = 0;
    for (int c = 0; c < nc; ++c) {
      const float s = out[(4 + c) * na + j];
      if (s > max_score) { max_score = s; max_class = c; }
    }
    if (max_score < conf_threshold) continue;

    dets.push_back({
      .x1 = row_cx[j] - row_w[j] * 0.5f, .y1 = row_cy[j] - row_h[j] * 0.5f,
      .x2 = row_cx[j] + row_w[j] * 0.5f, .y2 = row_cy[j] + row_h[j] * 0.5f,
      .score = max_score, .class_id = max_class
    });
  }
  return dets;
}

}  // namespace rtbot_yolo_trt
