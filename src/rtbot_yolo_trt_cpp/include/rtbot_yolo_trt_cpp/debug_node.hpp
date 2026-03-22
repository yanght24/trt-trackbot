// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// debug_node.hpp — v2: C++ 调试可视化节点（替代 Python debug_node.py）
///
/// 设计原则：不得拖慢主推理链路。
///   - 使用 ApproximateTimeSynchronizer 同步 image_raw + tracking
///   - 队列深度保持 1，始终丢弃旧帧——不累积积压
///   - 无订阅者时跳过绘制，避免浪费 CPU

#include "rtbot_yolo_trt_cpp/common.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <yolo_msgs/msg/detection_array.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace rtbot_yolo_trt
{

class DebugNode : public rclcpp::Node
{
public:
  explicit DebugNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  void syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & img_msg,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr & det_msg);

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
  std::shared_ptr<message_filters::Subscriber<yolo_msgs::msg::DetectionArray>> det_sub_;
  std::shared_ptr<Synchronizer> sync_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr dbg_pub_;

  // 按 class_id 缓存颜色，保证颜色稳定
  std::unordered_map<int, Color3> color_cache_;
};

}  // namespace rtbot_yolo_trt
