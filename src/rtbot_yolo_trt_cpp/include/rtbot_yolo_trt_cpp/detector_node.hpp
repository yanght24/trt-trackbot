// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// detector_node.hpp — v2: end2end TensorRT YOLO 检测 ROS 节点
///
/// 订阅图像，GPU 预处理 → TRT end2end 推理 → 发布 DetectionArray。
/// 无需 CPU NMS——EfficientNMS_TRT 插件已内嵌在 engine 中。

#include "rtbot_yolo_trt_cpp/trt_backend.hpp"
#include "rtbot_yolo_trt_cpp/common.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <yolo_msgs/msg/detection_array.hpp>

#include <memory>
#include <string>
#include <vector>

namespace rtbot_yolo_trt
{

class DetectorNode : public rclcpp::Node
{
public:
  explicit DetectorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void declareParams();
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  [[nodiscard]]
  yolo_msgs::msg::DetectionArray toMsg(
    const std::vector<Detection> & dets,
    const std_msgs::msg::Header & header) const;

  std::unique_ptr<TrtBackend> backend_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr det_pub_;

  // ROS 参数
  std::string engine_path_;
  std::string input_image_topic_;
  std::string detection_topic_;
  float conf_threshold_{0.25f};
  std::vector<std::string> class_names_;
};

}  // namespace rtbot_yolo_trt
