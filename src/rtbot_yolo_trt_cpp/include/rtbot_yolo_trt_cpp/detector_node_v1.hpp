// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// detector_node_v1.hpp — V1 检测节点：raw-head TrtEngine + CPU NMS
///
/// 使用标准 YOLO engine（[1,84,N] 输出），在 CPU 上执行 NMS。
/// 保留为独立编译目标，用于回归测试和与 V2 end2end 的性能对比。

#include "rtbot_yolo_trt_cpp/trt_engine.hpp"
#include "rtbot_yolo_trt_cpp/postprocess.hpp"
#include "rtbot_yolo_trt_cpp/common.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <yolo_msgs/msg/detection_array.hpp>

#include <memory>
#include <string>
#include <vector>

namespace rtbot_yolo_trt
{

class DetectorNodeV1 : public rclcpp::Node
{
public:
  explicit DetectorNodeV1(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void declareParams();
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  [[nodiscard]]
  yolo_msgs::msg::DetectionArray toMsg(
    const std::vector<Detection> & dets,
    const std_msgs::msg::Header & header) const;

  // Engine（v1 raw-head + CPU NMS）
  std::unique_ptr<TrtEngine> engine_;

  // ROS 接口
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr det_pub_;

  // ROS 参数
  std::string engine_path_;
  std::string input_image_topic_;
  std::string detection_topic_;
  double conf_threshold_{0.25};
  double iou_threshold_{0.65};
  int max_det_{100};
  std::vector<std::string> class_names_;
};

}  // namespace rtbot_yolo_trt
