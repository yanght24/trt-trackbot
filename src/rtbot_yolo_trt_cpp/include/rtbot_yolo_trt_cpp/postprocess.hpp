// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// postprocess.hpp — V1 后处理（CPU NMS + letterbox 坐标逆映射）

#include "rtbot_yolo_trt_cpp/common.hpp"
#include <vector>

namespace rtbot_yolo_trt
{

/// 将检测框从 letterbox 空间映射回原图坐标，
/// 按置信度过滤，执行每类 NMS，并限制最大检测数。
[[nodiscard]]
std::vector<Detection> postprocess(
  const std::vector<Detection> & raw_dets,
  const PreprocessInfo & info,
  float conf_threshold,
  float iou_threshold,
  int max_det);

}  // namespace rtbot_yolo_trt
