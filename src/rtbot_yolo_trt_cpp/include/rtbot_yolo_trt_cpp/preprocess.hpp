// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// preprocess.hpp — V1 CPU 预处理（letterbox + 归一化）
///
/// 提供持久化缓冲区（PreprocessBuffers），避免每帧堆分配。
/// V2 使用 GPU 预处理，此文件仅供 V1 的 CPU 预处理路径使用。

#include "rtbot_yolo_trt_cpp/common.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace rtbot_yolo_trt
{

/// 持久预处理缓冲区——初始化时分配一次，每帧复用。
/// 消除预处理管线中的所有逐帧堆分配。
struct PreprocessBuffers
{
  cv::Mat resized;     // resize 后
  cv::Mat padded;      // letterbox 填充后
  cv::Mat rgb;         // BGR→RGB 后
  cv::Mat float_img;   // convertTo float32 后
};

/// Letterbox + 归一化，写入预分配的 CHW float 缓冲区。
/// 所有中间 cv::Mat 保存在 bufs 中（无逐帧分配）。
/// @param src        输入 BGR 图像（任意尺寸）
/// @param target_w   模型输入宽度
/// @param target_h   模型输入高度
/// @param blob       输出 CHW float 缓冲区（须预分配 3*target_h*target_w）
/// @param bufs       可复用的中间缓冲区
/// @return           预处理元信息，用于后处理坐标逆映射
PreprocessInfo preprocess(
  const cv::Mat & src,
  int target_w,
  int target_h,
  float * blob,
  PreprocessBuffers & bufs);

}  // namespace rtbot_yolo_trt
