// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#include "rtbot_yolo_trt_cpp/detector_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <chrono>

namespace rtbot_yolo_trt
{

DetectorNode::DetectorNode(const rclcpp::NodeOptions & options)
: Node("detector_node", options)
{
  declareParams();

  // 加载 end2end TensorRT engine
  RCLCPP_INFO(get_logger(), "Loading end2end engine: %s", engine_path_.c_str());
  auto t0 = std::chrono::steady_clock::now();
  backend_ = std::make_unique<TrtBackend>(engine_path_);
  auto t1 = std::chrono::steady_clock::now();
  double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  RCLCPP_INFO(get_logger(), "Engine loaded in %.0f ms  (input: %dx%d)",
    load_ms, backend_->input_h(), backend_->input_w());

  // ROS 接口
  det_pub_ = create_publisher<yolo_msgs::msg::DetectionArray>(detection_topic_, 10);

  // 使用 RELIABLE QoS 以匹配 Gazebo 相机发布者（RELIABLE 发布者不兼容 BEST_EFFORT 订阅者）
  auto qos = rclcpp::QoS(1).reliable();
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    input_image_topic_, qos,
    std::bind(&DetectorNode::imageCallback, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(),
    "DetectorNode v2 ready. Sub: %s  Pub: %s  conf=%.2f",
    input_image_topic_.c_str(), detection_topic_.c_str(), conf_threshold_);
}

void DetectorNode::declareParams()
{
  engine_path_       = declare_parameter<std::string>("engine_path", "");
  input_image_topic_ = declare_parameter<std::string>("input_image_topic", "image_raw");
  detection_topic_   = declare_parameter<std::string>("detection_topic", "detections");
  conf_threshold_    = declare_parameter<double>("conf_threshold", 0.25);

  // 类别名称：默认 COCO 80 类
  class_names_ = declare_parameter<std::vector<std::string>>("class_names", cocoNames());

  if (engine_path_.empty()) {
    RCLCPP_FATAL(get_logger(), "Parameter 'engine_path' is required but not set.");
    throw std::runtime_error("engine_path not set");
  }
}

void DetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto t0 = std::chrono::steady_clock::now();

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge: %s", e.what());
    return;
  }

  const cv::Mat & raw_bgr = cv_ptr->image;

  // 确保内存连续以便 GPU 上传
  cv::Mat bgr;
  if (raw_bgr.isContinuous()) {
    bgr = raw_bgr;
  } else {
    bgr = raw_bgr.clone();
  }

  auto t1 = std::chrono::steady_clock::now();

  // End2end 推理: GPU 预处理 → TRT → 解析 end2end 输出
  // 不需要 CPU NMS——EfficientNMS_TRT 插件在 engine 内部完成
  auto result = backend_->infer(bgr.data, bgr.cols, bgr.rows, conf_threshold_);

  auto t2 = std::chrono::steady_clock::now();

  // 发布 DetectionArray（header 继承自原始图像）
  det_pub_->publish(toMsg(result.dets, msg->header));

  auto t3 = std::chrono::steady_clock::now();
  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
    "cvt=%.1f e2e_infer=%.1f pub=%.1f total=%.1f ms  dets=%zu",
    ms(t0,t1), ms(t1,t2), ms(t2,t3), ms(t0,t3), result.dets.size());
}

yolo_msgs::msg::DetectionArray DetectorNode::toMsg(
  const std::vector<Detection> & dets,
  const std_msgs::msg::Header & header) const
{
  yolo_msgs::msg::DetectionArray arr;
  arr.header = header;

  for (const auto & d : dets) {
    yolo_msgs::msg::Detection det_msg;
    det_msg.class_id = d.class_id;
    det_msg.score = d.score;

    // 类别名称查表，std::ssize() 返回有符号值，消除 static_cast<int> 转换
    if (d.class_id >= 0 && d.class_id < std::ssize(class_names_)) {
      det_msg.class_name = class_names_[d.class_id];
    } else {
      det_msg.class_name = "class_" + std::to_string(d.class_id);
    }

    // BoundingBox2D: 中心 + 尺寸格式（yolo_msgs 约定）
    const float cx = (d.x1 + d.x2) / 2.0f;
    const float cy = (d.y1 + d.y2) / 2.0f;
    const float w  = d.x2 - d.x1;
    const float h  = d.y2 - d.y1;

    det_msg.bbox.center.position.x = cx;
    det_msg.bbox.center.position.y = cy;
    det_msg.bbox.center.theta = 0.0;
    det_msg.bbox.size.x = w;
    det_msg.bbox.size.y = h;

    // id 字段留空（由 tracker_node 分配跟踪 ID）

    arr.detections.push_back(std::move(det_msg));
  }

  return arr;
}

}  // namespace rtbot_yolo_trt
