// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// common.hpp — v2 全栈共享的类型、常量和工具函数
///
/// detector / tracker / debug 三个节点均 include 此头文件。
/// v1 的 types.hpp 现已改为转发到本文件，消除重复定义。

#include <cstdint>
#include <string>
#include <vector>

namespace rtbot_yolo_trt
{

// ── 检测结果（原图坐标，NMS 后）────────────────────────────────────

struct Detection
{
  float x1, y1, x2, y2;   // 原图像素坐标的框角点
  float score;
  int   class_id;
};

// ── Letterbox 预处理元信息（用于坐标逆映射）──────────────────────

struct PreprocessInfo
{
  int   orig_w{0};
  int   orig_h{0};
  float scale{1.0f};      // letterbox 缩放比 = min(target_w/orig_w, target_h/orig_h)
  float pad_x{0.0f};      // 目标空间中的水平填充（像素）
  float pad_y{0.0f};      // 目标空间中的垂直填充（像素）
};

// ── COCO 80 类别名 ──────────────────────────────────────────────

inline const std::vector<std::string> & cocoNames()
{
  static const std::vector<std::string> names = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
  };
  return names;
}

// ── 按 ID 生成稳定颜色 ──────────────────────────────────────────

struct Color3 { uint8_t r, g, b; };

/// 为给定整数 key 生成视觉上有区分度的 BGR 颜色。
/// 使用黄金比例色调旋转，保证颜色间隔感良好。
inline Color3 colorForId(int id)
{
  const float h = static_cast<float>((id + 1) * 67 % 360) / 360.0f;  // 伪随机色调
  const float s = 0.7f + (id % 3) * 0.1f;
  const float v = 0.9f - (id % 5) * 0.05f;

  // HSV → RGB
  const int hi = static_cast<int>(h * 6.0f) % 6;
  const float f = h * 6.0f - static_cast<float>(hi);
  const float p = v * (1.0f - s);
  const float q = v * (1.0f - f * s);
  const float t = v * (1.0f - (1.0f - f) * s);

  float r, g, b;
  switch (hi) {
    case 0: r=v; g=t; b=p; break;
    case 1: r=q; g=v; b=p; break;
    case 2: r=p; g=v; b=t; break;
    case 3: r=p; g=q; b=v; break;
    case 4: r=t; g=p; b=v; break;
    default:r=v; g=p; b=q; break;
  }
  return {static_cast<uint8_t>(r*255), static_cast<uint8_t>(g*255), static_cast<uint8_t>(b*255)};
}

}  // namespace rtbot_yolo_trt
