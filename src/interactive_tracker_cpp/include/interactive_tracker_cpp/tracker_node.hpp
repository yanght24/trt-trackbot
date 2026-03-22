// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <yolo_msgs/msg/detection_array.hpp>

#include <deque>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace interactive_tracker_cpp
{

enum class TrackerState
{
  Manual,
  Locked,
  Searching
};

struct TrackedTarget
{
  std::string tracking_id;
  int class_id{0};
  std::string class_name;
  double center_x{0.0};
  double center_y{0.0};
  double width{0.0};
  double height{0.0};
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  int slot{-1};
};

class TrackerNode : public rclcpp::Node
{
public:
  explicit TrackerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void declareAndLoadParameters();
  void createInterfaces();

  void trackingCallback(const yolo_msgs::msg::DetectionArray::SharedPtr msg);
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void manualCmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void userCommandCallback(const std_msgs::msg::String::SharedPtr msg);
  void controlLoop();

  void updateSlotMapping(std::vector<TrackedTarget> & targets);

  std::optional<TrackedTarget> findTargetById(const std::string & tracking_id) const;
  std::optional<TrackedTarget> findNearestSameClass() const;
  geometry_msgs::msg::Twist computeTrackingCommand(const TrackedTarget & target);
  geometry_msgs::msg::Twist computeSearchCommand() const;
  geometry_msgs::msg::Twist zeroTwist() const;

  // LiDAR ROI distance extraction
  // extractRoiDistance: spatial filter (front-k mean) within angle window, handles wrap-around
  // getDistanceForTarget: maps target pixel pos -> LiDAR bearing, calls extractRoiDistance
  // updateDistanceFilter: applies temporal EMA on top of the spatial reading
  std::optional<double> extractRoiDistance(
    const sensor_msgs::msg::LaserScan & scan,
    double angle_center_rad,
    double angle_half_span_rad) const;
  std::optional<double> getDistanceForTarget(const TrackedTarget & target) const;
  double updateDistanceFilter(double new_reading);

  // Per-class LiDAR window overrides
  double getClassWindowScale(const std::string & class_name) const;
  double getClassMinWindowRad(const std::string & class_name) const;

  void publishTargetList();
  std::string buildTargetListJson() const;

  static std::string trim(const std::string & input);
  static std::string stateToString(TrackerState state);
  // Normalize angle to [0, 2π) to match scan frames where angle_min=0
  static double normalizeAngle2Pi(double angle);

  // ---------------------------------------------------------------------------
  // ROS interfaces
  // ---------------------------------------------------------------------------
  rclcpp::Subscription<yolo_msgs::msg::DetectionArray>::SharedPtr tracking_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr manual_cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr user_command_sub_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr locked_target_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr target_list_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_image_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr target_distance_pub_;

  rclcpp::TimerBase::SharedPtr control_timer_;

  // ---------------------------------------------------------------------------
  // Latest sensor data
  // ---------------------------------------------------------------------------
  std::vector<TrackedTarget> latest_targets_;
  geometry_msgs::msg::Twist latest_manual_cmd_;
  sensor_msgs::msg::Image::SharedPtr latest_image_;
  sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;

  // ---------------------------------------------------------------------------
  // FSM state
  // ---------------------------------------------------------------------------
  TrackerState state_{TrackerState::Manual};
  std::string locked_target_id_;
  std::string locked_class_name_;
  double last_known_center_x_{0.0};
  double last_known_center_y_{0.0};

  // Angular control filter state
  double filtered_center_x_{0.0};
  double prev_angular_z_{0.0};
  bool filter_initialized_{false};
  bool forward_allowed_{false};

  // LiDAR distance temporal filter state
  std::deque<double> distance_history_;   // sliding window of raw spatial readings
  double filtered_distance_m_{-1.0};     // EMA output (what control loop uses)
  bool dist_filter_initialized_{false};
  double last_measured_distance_m_{-1.0};  // same as filtered_distance_m_, for overlay/publish

  // ---------------------------------------------------------------------------
  // Stable slot mapping
  // ---------------------------------------------------------------------------
  std::map<int, std::string> slot_to_id_;
  std::map<std::string, int> id_to_slot_;
  std::map<std::string, rclcpp::Time> id_last_seen_;

  // ---------------------------------------------------------------------------
  // Timestamps
  // ---------------------------------------------------------------------------
  rclcpp::Time latest_manual_cmd_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_target_seen_time_{0, 0, RCL_ROS_TIME};

  // ---------------------------------------------------------------------------
  // Parameters — topics
  // ---------------------------------------------------------------------------
  std::string tracking_topic_;
  std::string manual_cmd_topic_;
  std::string user_command_topic_;
  std::string cmd_vel_topic_;
  std::string image_topic_;
  std::string scan_topic_;

  // ---------------------------------------------------------------------------
  // Parameters — angular (yaw) control
  // ---------------------------------------------------------------------------
  double image_center_x_{960.0};
  double image_width_{1920.0};
  double horizontal_fov_rad_{1.5708};
  double yaw_kp_{0.001};
  double yaw_deadband_px_{50.0};
  double center_x_alpha_{0.5};
  double angular_z_alpha_{0.7};
  double max_angular_z_{0.6};

  // ---------------------------------------------------------------------------
  // Parameters — linear (distance) control limits
  // ---------------------------------------------------------------------------
  double max_linear_x_{0.20};
  double max_reverse_x_{0.08};
  bool allow_reverse_{false};
  double forward_gate_open_px_{120.0};
  double forward_gate_close_px_{220.0};

  // ---------------------------------------------------------------------------
  // Parameters — LiDAR distance control (primary)
  // ---------------------------------------------------------------------------
  bool use_lidar_distance_{true};
  double desired_distance_m_{1.2};
  double distance_kp_{0.6};
  double distance_deadband_m_{0.10};
  double lidar_angle_window_scale_{1.0};
  double lidar_min_window_rad_{0.05};
  double lidar_max_distance_m_{8.0};
  // Camera-to-LiDAR yaw offset: add to camera bearing before querying scan
  // Positive = LiDAR is rotated CCW relative to camera
  double camera_lidar_yaw_offset_rad_{0.0};
  // Minimum valid points in ROI; fewer -> nullopt (don't trust sparse readings)
  int lidar_min_valid_points_{3};
  // Temporal filter: sliding window length and EMA alpha
  int distance_filter_window_{5};
  double distance_ema_alpha_{0.4};

  // ---------------------------------------------------------------------------
  // Parameters — bbox height fallback (secondary)
  // ---------------------------------------------------------------------------
  bool use_bbox_height_fallback_{true};
  double linear_kp_{0.002};
  double target_height_desired_{300.0};
  double bbox_height_deadband_{25.0};

  // ---------------------------------------------------------------------------
  // Parameters — slot / target management
  // ---------------------------------------------------------------------------
  double max_reacquire_dist_px_{300.0};
  double manual_cmd_timeout_sec_{0.25};
  double target_lost_timeout_sec_{0.60};
  double slot_release_timeout_sec_{2.0};
  double search_angular_z_{0.40};
  double control_rate_hz_{20.0};
};

}  // namespace interactive_tracker_cpp
