// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#include "rtbot_yolo_trt_cpp/tracker_node.hpp"

#include <algorithm>
#include <ranges>

namespace rtbot_yolo_trt
{

// ── 构造函数 ────────────────────────────────────────────────────────

TrackerNode::TrackerNode(const rclcpp::NodeOptions & options)
: Node("tracker_node", options)
{
  iou_threshold_ = declare_parameter<double>("iou_threshold", 0.3);
  max_age_       = declare_parameter<int>("max_age", 30);
  min_hits_      = declare_parameter<int>("min_hits", 1);

  det_sub_ = create_subscription<yolo_msgs::msg::DetectionArray>(
    "detections", 10,
    std::bind(&TrackerNode::detectionsCallback, this, std::placeholders::_1));

  trk_pub_ = create_publisher<yolo_msgs::msg::DetectionArray>("tracking", 10);

  RCLCPP_INFO(get_logger(),
    "TrackerNode ready. iou_thresh=%.2f max_age=%d min_hits=%d",
    iou_threshold_, max_age_, min_hits_);
}

// ── IoU 计算 ────────────────────────────────────────────────────────

float TrackerNode::iou(
  float ax1, float ay1, float ax2, float ay2,
  float bx1, float by1, float bx2, float by2)
{
  const float ix1 = std::max(ax1, bx1);
  const float iy1 = std::max(ay1, by1);
  const float ix2 = std::min(ax2, bx2);
  const float iy2 = std::min(ay2, by2);
  const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
  const float area_a = (ax2 - ax1) * (ay2 - ay1);
  const float area_b = (bx2 - bx1) * (by2 - by1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

// ── 主跟踪回调 ──────────────────────────────────────────────────────

void TrackerNode::detectionsCallback(const yolo_msgs::msg::DetectionArray::SharedPtr msg)
{
  const auto & detections = msg->detections;
  const int n_det = std::ssize(detections);   // C++20 std::ssize(): 直接返回有符号值，无需 static_cast
  const int n_trk = std::ssize(tracks_);

  // ── 第1步: 提取检测框 ──────────────────────────────────────────
  struct DetBox {
    float x1, y1, x2, y2, score;
    int class_id;
    int det_idx;
  };

  std::vector<DetBox> det_boxes;
  det_boxes.reserve(n_det);
  for (int i = 0; i < n_det; ++i) {
    const auto & d = detections[i];
    const float cx = static_cast<float>(d.bbox.center.position.x);
    const float cy = static_cast<float>(d.bbox.center.position.y);
    const float hw = static_cast<float>(d.bbox.size.x) / 2.0f;
    const float hh = static_cast<float>(d.bbox.size.y) / 2.0f;
    // C++20 designated initializer — 字段名直接对应结构体成员，无歧义
    det_boxes.push_back({
      .x1 = cx - hw, .y1 = cy - hh,
      .x2 = cx + hw, .y2 = cy + hh,
      .score = static_cast<float>(d.score),
      .class_id = d.class_id,
      .det_idx = i
    });
  }

  // C++20 ranges::sort + 投影：按 score 成员降序，不用写完整比较 lambda
  std::ranges::sort(det_boxes, std::greater{}, &DetBox::score);

  // ── 第2步: 贪心 IoU 匹配（检测 → 跟踪目标）────────────────────
  std::vector<bool> det_matched(n_det, false);
  std::vector<bool> trk_matched(n_trk, false);

  for (int di = 0; di < n_det; ++di) {
    const auto & db = det_boxes[di];
    float best_iou = iou_threshold_;
    int   best_ti  = -1;

    for (int ti = 0; ti < n_trk; ++ti) {
      if (trk_matched[ti]) [[unlikely]] continue;
      if (tracks_[ti].class_id != db.class_id) [[likely]] continue;  // 不同类别概率更高

      const float io = iou(db.x1, db.y1, db.x2, db.y2,
                           tracks_[ti].x1, tracks_[ti].y1,
                           tracks_[ti].x2, tracks_[ti].y2);
      if (io > best_iou) {
        best_iou = io;
        best_ti  = ti;
      }
    }

    if (best_ti >= 0) {
      det_matched[di]      = true;
      trk_matched[best_ti] = true;
      auto & t = tracks_[best_ti];
      t.x1    = db.x1;  t.y1 = db.y1;
      t.x2    = db.x2;  t.y2 = db.y2;
      t.score = db.score;
      t.age   = 0;
      t.hits++;
    }
  }

  // ── 第3步: 为未匹配的检测创建新跟踪目标 ────────────────────────
  for (int di = 0; di < n_det; ++di) {
    if (det_matched[di]) [[likely]] continue;
    const auto & db = det_boxes[di];
    // C++20 designated initializer — 明确每个字段的含义
    tracks_.push_back(Track{
      .id       = next_id_++,
      .class_id = db.class_id,
      .x1 = db.x1, .y1 = db.y1,
      .x2 = db.x2, .y2 = db.y2,
      .score = db.score,
      .age  = 0,
      .hits = 1
    });
  }

  // ── 第4步: 老化未匹配跟踪目标，移除过期的 ─────────────────────
  for (int ti = 0; ti < n_trk; ++ti) {
    if (!trk_matched[ti]) tracks_[ti].age++;
  }

  // C++20 std::erase_if — 替代 erase(remove_if(...), end()) 两步操作
  std::erase_if(tracks_, [this](const Track & t) { return t.age > max_age_; });

  // ── 第5步: 构建输出消息 ────────────────────────────────────────
  yolo_msgs::msg::DetectionArray out;
  out.header = msg->header;

  for (const auto & t : tracks_) {
    if (t.hits < min_hits_) [[likely]] continue;
    if (t.age > 0) [[likely]] continue;

    yolo_msgs::msg::Detection det_msg;
    det_msg.class_id = t.class_id;
    det_msg.score    = t.score;
    det_msg.id       = std::to_string(t.id);

    // 按 class_id 查找类别名称
    for (int di = 0; di < n_det; ++di) {
      if (det_boxes[di].class_id == t.class_id) [[unlikely]] {
        det_msg.class_name = detections[det_boxes[di].det_idx].class_name;
        break;
      }
    }
    if (det_msg.class_name.empty()) {
      det_msg.class_name = "class_" + std::to_string(t.class_id);
    }

    const float cx = (t.x1 + t.x2) / 2.0f;
    const float cy = (t.y1 + t.y2) / 2.0f;
    det_msg.bbox.center.position.x = cx;
    det_msg.bbox.center.position.y = cy;
    det_msg.bbox.center.theta = 0.0;
    det_msg.bbox.size.x = t.x2 - t.x1;
    det_msg.bbox.size.y = t.y2 - t.y1;

    out.detections.push_back(std::move(det_msg));
  }

  trk_pub_->publish(out);
}

}  // namespace rtbot_yolo_trt
