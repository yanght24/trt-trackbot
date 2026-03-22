// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// cuda_preprocess.hpp — GPU 加速的 letterbox + BGR→RGB + 归一化 + HWC→CHW
///
/// 输入:  d_src  — device 指针，uint8 BGR 图像，HWC 布局 [src_h, src_w, 3]
/// 输出:  d_dst  — device 指针，float32 RGB 图像，CHW 布局 [3, dst_h, dst_w]，归一化到 [0,1]
///
/// kernel 执行双线性插值 resize + letterbox 填充（填充值 114/255）。
/// 不需要任何 host 端中间缓冲区。

#include <cuda_runtime.h>
#include <cstdint>

namespace rtbot_yolo_trt
{

/// @param d_src     源 uint8 BGR 图像的 device 指针
/// @param src_w     源图宽度
/// @param src_h     源图高度
/// @param d_dst     目标 float32 CHW 缓冲区的 device 指针（须预分配）
/// @param dst_w     目标（模型输入）宽度
/// @param dst_h     目标（模型输入）高度
/// @param pad_x     目标空间中的水平填充（像素）
/// @param pad_y     目标空间中的垂直填充（像素）
/// @param scale     letterbox 缩放因子
/// @param stream    CUDA 流，用于异步执行
void cudaPreprocessLetterbox(
  const uint8_t * d_src, int src_w, int src_h,
  float * d_dst, int dst_w, int dst_h,
  float pad_x, float pad_y, float scale,
  cudaStream_t stream);

}  // namespace rtbot_yolo_trt
