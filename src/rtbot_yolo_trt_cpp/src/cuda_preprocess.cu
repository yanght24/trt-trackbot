// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#include "rtbot_yolo_trt_cpp/cuda_preprocess.hpp"

#include <cstdio>
#include <stdexcept>

namespace rtbot_yolo_trt
{

// 每个线程处理目标（letterbox）图像中的一个像素。
// 落在缩放源区域内的像素：从源图双线性采样，BGR→RGB，归一化。
// 填充区域的像素：填充 114/255 ≈ 0.447。
__global__ void letterboxKernel(
  const uint8_t * __restrict__ src, int src_w, int src_h,
  float * __restrict__ dst, int dst_w, int dst_h,
  float pad_x, float pad_y, float scale)
{
  const int dx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dy = blockIdx.y * blockDim.y + threadIdx.y;

  if (dx >= dst_w || dy >= dst_h) return;

  const int hw = dst_h * dst_w;
  const float pad_val = 114.0f / 255.0f;

  // 将目标像素中心映射到源图坐标。
  // +0.5 / -0.5 对齐像素中心（避免边缘半像素偏移）。
  const float sx = ((dx + 0.5f) - pad_x) / scale - 0.5f;
  const float sy = ((dy + 0.5f) - pad_y) / scale - 0.5f;

  float r, g, b;

  if (sx < -0.5f || sx >= (src_w - 0.5f) || sy < -0.5f || sy >= (src_h - 0.5f)) {
    r = g = b = pad_val;
  } else {
    // 从源图（BGR uint8）双线性插值
    const float sx_c = fmaxf(0.0f, fminf(sx, src_w - 1.001f));
    const float sy_c = fmaxf(0.0f, fminf(sy, src_h - 1.001f));
    const int x0 = static_cast<int>(sx_c);
    const int y0 = static_cast<int>(sy_c);
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);

    const float fx = sx_c - x0;
    const float fy = sy_c - y0;
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    const int stride = src_w * 3;

    #define FETCH(Y, X, C) (static_cast<float>(src[(Y) * stride + (X) * 3 + (C)]))

    // BGR 源 → RGB 输出：源通道 2=R, 1=G, 0=B
    b = (w00*FETCH(y0,x0,0) + w01*FETCH(y0,x1,0) + w10*FETCH(y1,x0,0) + w11*FETCH(y1,x1,0)) / 255.0f;
    g = (w00*FETCH(y0,x0,1) + w01*FETCH(y0,x1,1) + w10*FETCH(y1,x0,1) + w11*FETCH(y1,x1,1)) / 255.0f;
    r = (w00*FETCH(y0,x0,2) + w01*FETCH(y0,x1,2) + w10*FETCH(y1,x0,2) + w11*FETCH(y1,x1,2)) / 255.0f;

    #undef FETCH
  }

  // 写入 CHW RGB 格式
  dst[0 * hw + dy * dst_w + dx] = r;
  dst[1 * hw + dy * dst_w + dx] = g;
  dst[2 * hw + dy * dst_w + dx] = b;
}

void cudaPreprocessLetterbox(
  const uint8_t * d_src, int src_w, int src_h,
  float * d_dst, int dst_w, int dst_h,
  float pad_x, float pad_y, float scale,
  cudaStream_t stream)
{
  const dim3 block(16, 16);
  const dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);

  letterboxKernel<<<grid, block, 0, stream>>>(
    d_src, src_w, src_h,
    d_dst, dst_w, dst_h,
    pad_x, pad_y, scale);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    char msg[256];
    snprintf(msg, sizeof(msg), "letterboxKernel launch failed: %s", cudaGetErrorString(err));
    throw std::runtime_error(msg);
  }
}

}  // namespace rtbot_yolo_trt
