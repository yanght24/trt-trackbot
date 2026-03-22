// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#include "rtbot_yolo_trt_cpp/detector_node_v1.hpp"
#include "rtbot_yolo_trt_cpp/common.hpp"      // cocoNames()
#include "rtbot_yolo_trt_cpp/preprocess.hpp"

#include <cv_bridge/cv_bridge.h>
#include <chrono>

namespace rtbot_yolo_trt
{

DetectorNodeV1::DetectorNodeV1(const rclcpp::NodeOptions & options)
: Node("detector_node_v1", options)
{
  declareParams();

  RCLCPP_INFO(get_logger(), "Loading V1 raw-head engine: %s", engine_path_.c_str());
  auto t0 = std::chrono::steady_clock::now();
  engine_ = std::make_unique<TrtEngine>(engine_path_);
  auto t1 = std::chrono::steady_clock::now();
  double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  RCLCPP_INFO(get_logger(), "V1 engine loaded in %.0f ms  (input: %dx%d  classes: %d  anchors: %d)",
    load_ms, engine_->input_h(), engine_->input_w(),
    engine_->num_classes(), engine_->num_anchors());

  det_pub_ = create_publisher<yolo_msgs::msg::DetectionArray>(detection_topic_, 10);

  // 使用 RELIABLE QoS 以匹配 Gazebo 相机发布者
  auto qos = rclcpp::QoS(1).reliable();
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    input_image_topic_, qos,
    std::bind(&DetectorNodeV1::imageCallback, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(),
    "DetectorNodeV1 ready. Sub: %s  Pub: %s  conf=%.2f  iou=%.2f  max_det=%d",
    input_image_topic_.c_str(), detection_topic_.c_str(),
    conf_threshold_, iou_threshold_, max_det_);
}

void DetectorNodeV1::declareParams()
{
  engine_path_       = declare_parameter<std::string>("engine_path", "");
  input_image_topic_ = declare_parameter<std::string>("input_image_topic", "image_raw");
  detection_topic_   = declare_parameter<std::string>("detection_topic", "detections");
  conf_threshold_    = declare_parameter<double>("conf_threshold", 0.25);
  iou_threshold_     = declare_parameter<double>("iou_threshold", 0.65);
  max_det_           = declare_parameter<int>("max_det", 100);
  class_names_       = declare_parameter<std::vector<std::string>>("class_names", cocoNames());

  if (engine_path_.empty()) {
    RCLCPP_FATAL(get_logger(), "Parameter 'engine_path' is required but not set.");
    throw std::runtime_error("engine_path not set");
  }
}

void DetectorNodeV1::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
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
  cv::Mat bgr;
  if (raw_bgr.isContinuous()) {
    bgr = raw_bgr;
  } else {
    bgr = raw_bgr.clone();
  }

  auto t1 = std::chrono::steady_clock::now();

  // V1 路径: GPU 预处理 → raw-head 推理 → CPU NMS
  auto raw_dets = engine_->inferWithGpuPreprocess(bgr.data, bgr.cols, bgr.rows, conf_threshold_);

  auto t2 = std::chrono::steady_clock::now();

  // 计算 letterbox 参数，用于后处理坐标映射
  const float r_w = static_cast<float>(engine_->input_w()) / bgr.cols;
  const float r_h = static_cast<float>(engine_->input_h()) / bgr.rows;
  PreprocessInfo info;
  info.orig_w = bgr.cols;
  info.orig_h = bgr.rows;
  info.scale  = std::min(r_w, r_h);
  info.pad_x  = (engine_->input_w() - bgr.cols * info.scale) / 2.0f;
  info.pad_y  = (engine_->input_h() - bgr.rows * info.scale) / 2.0f;

  auto final_dets = postprocess(raw_dets, info, conf_threshold_, iou_threshold_, max_det_);

  auto t3 = std::chrono::steady_clock::now();

  det_pub_->publish(toMsg(final_dets, msg->header));

  auto t4 = std::chrono::steady_clock::now();

  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
    "cvt=%.1f infer=%.1f nms=%.1f pub=%.1f total=%.1f ms  raw=%zu final=%zu",
    ms(t0,t1), ms(t1,t2), ms(t2,t3), ms(t3,t4), ms(t0,t4),
    raw_dets.size(), final_dets.size());
}

yolo_msgs::msg::DetectionArray DetectorNodeV1::toMsg(
  const std::vector<Detection> & dets,
  const std_msgs::msg::Header & header) const
{
  yolo_msgs::msg::DetectionArray arr;
  arr.header = header;

  for (const auto & d : dets) {
    yolo_msgs::msg::Detection det_msg;
    det_msg.class_id = d.class_id;
    det_msg.score = d.score;

    if (d.class_id >= 0 && d.class_id < std::ssize(class_names_)) {
      det_msg.class_name = class_names_[d.class_id];
    } else {
      det_msg.class_name = "class_" + std::to_string(d.class_id);
    }

    const float cx = (d.x1 + d.x2) / 2.0f;
    const float cy = (d.y1 + d.y2) / 2.0f;
    const float w  = d.x2 - d.x1;
    const float h  = d.y2 - d.y1;

    det_msg.bbox.center.position.x = cx;
    det_msg.bbox.center.position.y = cy;
    det_msg.bbox.center.theta = 0.0;
    det_msg.bbox.size.x = w;
    det_msg.bbox.size.y = h;

    arr.detections.push_back(std::move(det_msg));
  }

  return arr;
}

}  // namespace rtbot_yolo_trt
