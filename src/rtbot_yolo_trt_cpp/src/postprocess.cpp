// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#include "rtbot_yolo_trt_cpp/postprocess.hpp"

#include <algorithm>
#include <cmath>

namespace rtbot_yolo_trt
{

// NMS 用的 IoU 计算
static float iou(const Detection & a, const Detection & b)
{
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

// 每类 NMS（贪心，按置信度降序）
static std::vector<Detection> nms(
  std::vector<Detection> & dets, float iou_threshold)
{
  // C++20 ranges::sort + 投影：直接按 score 成员降序，无需 lambda
  std::ranges::sort(dets, std::greater{}, &Detection::score);

  std::vector<bool> suppressed(dets.size(), false);
  std::vector<Detection> result;
  result.reserve(dets.size());

  for (size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) [[unlikely]] continue;   // 大多数框不会被抑制
    result.push_back(dets[i]);
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (suppressed[j]) [[likely]] continue;   // 已抑制的框概率较高
      if (dets[i].class_id == dets[j].class_id && iou(dets[i], dets[j]) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return result;
}

std::vector<Detection> postprocess(
  const std::vector<Detection> & raw_dets,
  const PreprocessInfo & info,
  float conf_threshold,
  float iou_threshold,
  int max_det)
{
  // 1. 按置信度过滤
  std::vector<Detection> filtered;
  filtered.reserve(raw_dets.size());
  for (const auto & d : raw_dets) {
    if (d.score >= conf_threshold) [[unlikely]] filtered.push_back(d);  // 大多数 anchor 会被过滤掉
  }

  // 2. NMS
  auto nms_result = nms(filtered, iou_threshold);

  // 3. letterbox 空间 → 原图坐标
  for (auto & d : nms_result) {
    d.x1 = std::clamp((d.x1 - info.pad_x) / info.scale, 0.0f, static_cast<float>(info.orig_w));
    d.y1 = std::clamp((d.y1 - info.pad_y) / info.scale, 0.0f, static_cast<float>(info.orig_h));
    d.x2 = std::clamp((d.x2 - info.pad_x) / info.scale, 0.0f, static_cast<float>(info.orig_w));
    d.y2 = std::clamp((d.y2 - info.pad_y) / info.scale, 0.0f, static_cast<float>(info.orig_h));
  }

  // 4. 移除退化框 — C++20 std::erase_if 替代 erase-remove 惯用法
  std::erase_if(nms_result, [](const Detection & d) {
    return (d.x2 - d.x1) < 1.0f || (d.y2 - d.y1) < 1.0f;
  });

  // 5. 限制最大检测数
  if (std::ssize(nms_result) > max_det) {
    nms_result.resize(max_det);
  }

  return nms_result;
}

}  // namespace rtbot_yolo_trt
