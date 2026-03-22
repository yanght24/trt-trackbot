// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#include "rtbot_yolo_trt_cpp/preprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>

namespace rtbot_yolo_trt
{

PreprocessInfo preprocess(
  const cv::Mat & src,
  int target_w,
  int target_h,
  float * blob,
  PreprocessBuffers & bufs)
{
  PreprocessInfo info;
  info.orig_w = src.cols;
  info.orig_h = src.rows;

  const float r_w = static_cast<float>(target_w) / src.cols;
  const float r_h = static_cast<float>(target_h) / src.rows;
  info.scale = std::min(r_w, r_h);

  const int new_w = static_cast<int>(src.cols * info.scale);
  const int new_h = static_cast<int>(src.rows * info.scale);
  info.pad_x = (target_w - new_w) / 2.0f;
  info.pad_y = (target_h - new_h) / 2.0f;

  // Resize（bufs.resized 跨帧复用——首帧后无分配）
  cv::resize(src, bufs.resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  // Letterbox 填充——复用 bufs.padded，仅在尺寸变化时重建
  if (bufs.padded.rows != target_h || bufs.padded.cols != target_w) {
    bufs.padded.create(target_h, target_w, CV_8UC3);
  }
  bufs.padded.setTo(cv::Scalar(114, 114, 114));
  bufs.resized.copyTo(bufs.padded(cv::Rect(
    static_cast<int>(info.pad_x), static_cast<int>(info.pad_y), new_w, new_h)));

  // BGR → RGB
  cv::cvtColor(bufs.padded, bufs.rgb, cv::COLOR_BGR2RGB);

  // 归一化到 float32 [0, 1]——单次操作，OpenCV 内部 SIMD 矢量化
  bufs.rgb.convertTo(bufs.float_img, CV_32FC3, 1.0 / 255.0);

  // HWC → CHW: 直接写入输出 blob（pinned memory）。
  // 避免 cv::split（会分配 3 个临时 Mat）和 3 次 memcpy。
  // 改为单次遍历 float 图像。
  const int hw = target_h * target_w;
  const float * src_ptr = reinterpret_cast<const float *>(bufs.float_img.data);
  float * ch_r = blob;
  float * ch_g = blob + hw;
  float * ch_b = blob + 2 * hw;

  for (int i = 0; i < hw; ++i) {
    ch_r[i] = src_ptr[i * 3 + 0];
    ch_g[i] = src_ptr[i * 3 + 1];
    ch_b[i] = src_ptr[i * 3 + 2];
  }

  return info;
}

}  // namespace rtbot_yolo_trt
