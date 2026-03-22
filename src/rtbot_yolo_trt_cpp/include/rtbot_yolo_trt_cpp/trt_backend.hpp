// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// trt_backend.hpp — end2end TensorRT YOLO 推理后端 (v2) 的 RAII 封装
///
/// End2end engine IO 张量布局：
///   输入:  "images"       [1, 3, H, W]         float32
///   输出:  "num"          [1]                   int32   — 检测数量
///          "boxes"        [1, max_det, 4]       float32 — x1,y1,x2,y2 (letterbox 空间)
///          "scores"       [1, max_det]          float32
///          "classes"      [1, max_det]          int32
///
/// 注: v1 使用 raw-head [1,84,N] 输出 + CPU NMS（见 trt_engine.hpp）。
///     v1 代码保留用于对比基准测试，v2 launch 仅使用本后端。

#include "rtbot_yolo_trt_cpp/common.hpp"
#include "rtbot_yolo_trt_cpp/logger.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace rtbot_yolo_trt
{

class TrtBackend
{
public:
  explicit TrtBackend(const std::string & engine_path);
  ~TrtBackend();

  TrtBackend(const TrtBackend &) = delete;
  TrtBackend & operator=(const TrtBackend &) = delete;

  /// 对原始 BGR uint8 图像执行 GPU 预处理 + 推理。
  /// 返回的检测框坐标已映射回原图空间。
  struct InferResult {
    std::vector<Detection> dets;   // 原图坐标
    PreprocessInfo         prep;   // letterbox 元信息
  };

  [[nodiscard]]
  InferResult infer(const uint8_t * h_bgr, int src_w, int src_h,
                    float conf_threshold = 0.25f);

  [[nodiscard]] int input_h() const { return input_h_; }
  [[nodiscard]] int input_w() const { return input_w_; }

  // RAII 释放器（public 以便文件作用域辅助函数使用）
  struct CudaDeleter  { void operator()(void * p) const { if (p) cudaFree(p); } };
  struct PinnedDeleter { void operator()(void * p) const { if (p) cudaFreeHost(p); } };

private:
  void discoverTensors();   // 自动发现 engine IO 张量名称和形状
  void allocateBuffers();   // 分配 device/pinned 缓冲区

  TrtLogger logger_;

  struct TrtDeleter {
    void operator()(nvinfer1::IRuntime * p) const          { delete p; }
    void operator()(nvinfer1::ICudaEngine * p) const       { delete p; }
    void operator()(nvinfer1::IExecutionContext * p) const  { delete p; }
  };

  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>          runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>       engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>  context_;

  cudaStream_t stream_{nullptr};

  // Device 缓冲区
  std::unique_ptr<void, CudaDeleter> d_input_;       // [1,3,H,W] float
  std::unique_ptr<void, CudaDeleter> d_num_;          // [1] int32
  std::unique_ptr<void, CudaDeleter> d_boxes_;        // [1,max_det,4] float
  std::unique_ptr<void, CudaDeleter> d_scores_;       // [1,max_det] float
  std::unique_ptr<void, CudaDeleter> d_classes_;      // [1,max_det] int32
  std::unique_ptr<void, CudaDeleter> d_src_img_;      // 源图像 BGR uint8（GPU 预处理用）

  // Pinned host 缓冲区（异步 DMA 用）
  std::unique_ptr<void, PinnedDeleter> h_src_pinned_;
  std::unique_ptr<void, PinnedDeleter> h_num_pinned_;
  std::unique_ptr<void, PinnedDeleter> h_boxes_pinned_;
  std::unique_ptr<void, PinnedDeleter> h_scores_pinned_;
  std::unique_ptr<void, PinnedDeleter> h_classes_pinned_;

  size_t src_pinned_capacity_{0};
  size_t d_src_capacity_{0};

  // 张量名称
  std::string name_input_;
  std::string name_num_;
  std::string name_boxes_;
  std::string name_scores_;
  std::string name_classes_;

  // 形状信息
  int input_h_{0};
  int input_w_{0};
  size_t input_size_bytes_{0};

  int max_det_{100};
  size_t num_bytes_{0};
  size_t boxes_bytes_{0};
  size_t scores_bytes_{0};
  size_t classes_bytes_{0};
};

}  // namespace rtbot_yolo_trt
