// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#include "rtbot_yolo_trt_cpp/debug_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

namespace rtbot_yolo_trt
{

DebugNode::DebugNode(const rclcpp::NodeOptions & options)
: Node("debug_node", options)
{
  // QoS: RELIABLE 匹配 Gazebo 相机发布者，depth=1 只保留最新帧
  rmw_qos_profile_t img_qos = rmw_qos_profile_default;
  img_qos.depth = 1;

  image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
    this, "image_raw", img_qos);
  det_sub_ = std::make_shared<message_filters::Subscriber<yolo_msgs::msg::DetectionArray>>(
    this, "tracking", rmw_qos_profile_default);

  // 近似时间同步器，小队列——丢弃旧数据而非累积
  sync_ = std::make_shared<Synchronizer>(SyncPolicy(5), *image_sub_, *det_sub_);
  sync_->registerCallback(
    std::bind(&DebugNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

  dbg_pub_ = create_publisher<sensor_msgs::msg::Image>("dbg_image", 10);

  RCLCPP_INFO(get_logger(), "DebugNode v2 ready. Sub: image_raw + tracking -> Pub: dbg_image");
}

void DebugNode::syncCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & img_msg,
  const yolo_msgs::msg::DetectionArray::ConstSharedPtr & det_msg)
{
  // 无订阅者时跳过——避免无人查看时浪费 CPU
  if (dbg_pub_->get_subscription_count() == 0) return;

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(img_msg, "bgr8");
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge: %s", e.what());
    return;
  }

  cv::Mat canvas = cv_ptr->image.clone();

  for (const auto & det : det_msg->detections) {
    const int class_id = det.class_id;

    // 按 class_id 查找/生成稳定颜色
    auto it = color_cache_.find(class_id);
    if (it == color_cache_.end()) {
      it = color_cache_.emplace(class_id, colorForId(class_id)).first;
    }
    const auto & c = it->second;
    const cv::Scalar color_bgr(c.b, c.g, c.r);

    // 从中心+尺寸还原框角点
    const float cx = det.bbox.center.position.x;
    const float cy = det.bbox.center.position.y;
    const float w  = det.bbox.size.x;
    const float h  = det.bbox.size.y;
    const int x1 = static_cast<int>(cx - w / 2.0);
    const int y1 = static_cast<int>(cy - h / 2.0);
    const int x2 = static_cast<int>(cx + w / 2.0);
    const int y2 = static_cast<int>(cy + h / 2.0);

    // 绘制边界框
    cv::rectangle(canvas, cv::Point(x1, y1), cv::Point(x2, y2), color_bgr, 2);

    // 构建标签: "类别名 (跟踪ID) 置信度"
    std::string label = det.class_name;
    if (!det.id.empty()) {
      label += " (" + det.id + ")";
    }
    char score_buf[16];
    std::snprintf(score_buf, sizeof(score_buf), " %.2f", det.score);
    label += score_buf;

    // 绘制标签背景
    int baseline = 0;
    const auto text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
    const int label_y = std::max(y1, text_size.height + 4);
    cv::rectangle(canvas,
      cv::Point(x1, label_y - text_size.height - 4),
      cv::Point(x1 + text_size.width + 4, label_y + 2),
      color_bgr, cv::FILLED);

    // 绘制标签文字（白字彩底）
    cv::putText(canvas, label,
      cv::Point(x1 + 2, label_y - 2),
      cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  }

  // 发布调试图像
  auto out_msg = cv_bridge::CvImage(img_msg->header, "bgr8", canvas).toImageMsg();
  dbg_pub_->publish(*out_msg);
}

}  // namespace rtbot_yolo_trt
