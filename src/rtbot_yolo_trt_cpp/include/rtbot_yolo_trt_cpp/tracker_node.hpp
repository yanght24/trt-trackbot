// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// tracker_node.hpp — v2: C++ 多目标跟踪节点（替代 Python tracking_node.py）
///
/// 简化版贪心 IoU 跟踪器。无卡尔曼滤波，无运动预测。
/// 优先目标：稳定、接口兼容，使 interactive_tracker_cpp 无需修改即可正常工作。

#include <rclcpp/rclcpp.hpp>
#include <yolo_msgs/msg/detection_array.hpp>

#include <string>
#include <vector>
#include <unordered_map>

namespace rtbot_yolo_trt
{

class TrackerNode : public rclcpp::Node
{
public:
  explicit TrackerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void detectionsCallback(const yolo_msgs::msg::DetectionArray::SharedPtr msg);

  // ── 内部跟踪目标表示 ──────────────────────────────────────────
  struct Track {
    int    id;               // 全局唯一跟踪 ID
    int    class_id;
    float  x1, y1, x2, y2;  // 最近一次已知的边界框
    float  score;
    int    age{0};           // 自上次更新以来的帧数
    int    hits{0};          // 累计匹配帧数
  };

  /// 两个框的 IoU
  [[nodiscard]]
  static float iou(float ax1, float ay1, float ax2, float ay2,
                   float bx1, float by1, float bx2, float by2);

  // 活跃跟踪目标列表
  std::vector<Track> tracks_;

  // ID 计数器——单调递增，与 ByteTrack 行为一致
  int next_id_{1};

  // ROS 参数
  float iou_threshold_{0.3f};   // 检测↔跟踪目标的最小 IoU 匹配阈值
  int   max_age_{30};           // 未匹配多少帧后删除（30 @ 30fps = 1秒）
  int   min_hits_{3};           // 至少命中几帧才发布（抗抖动）

  rclcpp::Subscription<yolo_msgs::msg::DetectionArray>::SharedPtr det_sub_;
  rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr    trk_pub_;
};

}  // namespace rtbot_yolo_trt
