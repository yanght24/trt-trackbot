// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// trt_engine.hpp — V1 标准 YOLO TensorRT 引擎的 RAII 封装
///
/// 支持标准 ultralytics YOLO 输出格式：
///   输入:  "images"   [1, 3, H, W]        float32
///   输出:  "output0"  [1, 84, num_anchors] float32
///     其中 84 = 4 (cx, cy, w, h) + 80 (类别得分)
///
/// NMS 在 CPU 端由 postprocess 执行。

#include "rtbot_yolo_trt_cpp/logger.hpp"
#include "rtbot_yolo_trt_cpp/common.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace rtbot_yolo_trt
{

class TrtEngine
{
public:
  explicit TrtEngine(const std::string & engine_path);
  ~TrtEngine();

  TrtEngine(const TrtEngine &) = delete;
  TrtEngine & operator=(const TrtEngine &) = delete;

  /// 对预处理后的 CHW float 缓冲区执行推理（CPU 预处理路径）
  [[nodiscard]]
  std::vector<Detection> infer(const float * input_chw, float conf_threshold = 0.25f);

  /// GPU 预处理推理：上传原始 BGR uint8 图像，在 GPU 上完成 letterbox+归一化。
  /// 快速路径——无需 CPU 预处理。
  [[nodiscard]]
  std::vector<Detection> inferWithGpuPreprocess(
    const uint8_t * h_bgr, int src_w, int src_h,
    float conf_threshold = 0.25f);

  [[nodiscard]] int input_h() const { return input_h_; }
  [[nodiscard]] int input_w() const { return input_w_; }
  [[nodiscard]] int input_c() const { return input_c_; }
  [[nodiscard]] int num_classes() const { return num_classes_; }
  [[nodiscard]] int num_anchors() const { return num_anchors_; }

  // RAII 释放器
  struct CudaDeleter { void operator()(void * p) const { if (p) cudaFree(p); } };
  struct PinnedDeleter { void operator()(void * p) const { if (p) cudaFreeHost(p); } };

  /// 获取 pinned host 输入缓冲区指针（直接写入预处理数据）
  float * pinned_input() { return static_cast<float *>(h_input_pinned_.get()); }
  size_t input_size_bytes() const { return input_size_bytes_; }

private:
  void allocateBuffers();

  TrtLogger logger_;

  struct TrtDeleter {
    void operator()(nvinfer1::IRuntime * p) const     { delete p; }
    void operator()(nvinfer1::ICudaEngine * p) const  { delete p; }
    void operator()(nvinfer1::IExecutionContext * p) const { delete p; }
  };

  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_;

  cudaStream_t stream_{nullptr};

  // Device 缓冲区
  std::unique_ptr<void, CudaDeleter> d_input_;      // float32 CHW [1,3,H,W]
  std::unique_ptr<void, CudaDeleter> d_output_;     // float32 raw 输出
  std::unique_ptr<void, CudaDeleter> d_src_img_;    // uint8 BGR 源图（GPU 预处理用）

  // Pinned host 缓冲区
  std::unique_ptr<void, PinnedDeleter> h_input_pinned_;  // CPU 预处理路径
  std::unique_ptr<void, PinnedDeleter> h_src_pinned_;    // GPU 预处理路径（原始 BGR 上传）
  std::unique_ptr<void, PinnedDeleter> h_output_pinned_; // D2H 输出（pinned 以支持异步 DMA）
  size_t src_pinned_capacity_{0};
  size_t d_src_capacity_{0};

  // 输出的类型化访问器（指向 h_output_pinned_）
  float * h_output_ptr_{nullptr};

  // Host 端输出大小
  size_t output_num_floats_{0};

  // 张量名称
  std::string name_input_;
  std::string name_output_;

  // 形状信息
  int input_c_{3};
  int input_h_{0};
  int input_w_{0};
  size_t input_size_bytes_{0};

  int num_classes_{80};    // 84 - 4
  int num_anchors_{0};     // anchor 点总数
  size_t output_size_bytes_{0};
};

}  // namespace rtbot_yolo_trt
