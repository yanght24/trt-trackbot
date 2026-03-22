// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#include "interactive_tracker_cpp/tracker_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace interactive_tracker_cpp
{

using namespace std::chrono_literals;

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

TrackerNode::TrackerNode(const rclcpp::NodeOptions & options)
: Node("interactive_tracker", options)
{
  declareAndLoadParameters();
  createInterfaces();

  RCLCPP_INFO(
    get_logger(),
    "interactive_tracker started. tracking=%s scan=%s image=%s cmd_vel=%s use_lidar=%s",
    tracking_topic_.c_str(), scan_topic_.c_str(), image_topic_.c_str(),
    cmd_vel_topic_.c_str(), use_lidar_distance_ ? "true" : "false");
}

// ---------------------------------------------------------------------------
// Parameter loading
// ---------------------------------------------------------------------------

void TrackerNode::declareAndLoadParameters()
{
  // -- Topics --
  tracking_topic_      = declare_parameter<std::string>("tracking_topic",      "/yolo/tracking");
  manual_cmd_topic_    = declare_parameter<std::string>("manual_cmd_topic",    "/manual_cmd_vel");
  user_command_topic_  = declare_parameter<std::string>("user_command_topic",  "/user_command");
  cmd_vel_topic_       = declare_parameter<std::string>("cmd_vel_topic",       "/cmd_vel");
  image_topic_         = declare_parameter<std::string>("image_topic",         "/yolo/dbg_image");
  scan_topic_          = declare_parameter<std::string>("scan_topic",          "/scan");

  // -- Angular control --
  image_center_x_       = declare_parameter<double>("image_center_x",       960.0);
  image_width_          = declare_parameter<double>("image_width",           1920.0);
  horizontal_fov_rad_   = declare_parameter<double>("horizontal_fov_rad",   1.5708);
  yaw_kp_               = declare_parameter<double>("yaw_kp",               0.001);
  yaw_deadband_px_      = declare_parameter<double>("yaw_deadband_px",      50.0);
  center_x_alpha_       = declare_parameter<double>("center_x_alpha",       0.5);
  angular_z_alpha_      = declare_parameter<double>("angular_z_alpha",      0.7);
  max_angular_z_        = declare_parameter<double>("max_angular_z",        0.6);

  // -- Linear limits --
  max_linear_x_         = declare_parameter<double>("max_linear_x",         0.20);
  max_reverse_x_        = declare_parameter<double>("max_reverse_x",        0.08);
  allow_reverse_        = declare_parameter<bool>  ("allow_reverse",        false);

  // -- Forward hysteresis gate --
  forward_gate_open_px_  = declare_parameter<double>("forward_gate_open_px",  120.0);
  forward_gate_close_px_ = declare_parameter<double>("forward_gate_close_px", 220.0);

  // -- LiDAR distance control (primary) --
  use_lidar_distance_           = declare_parameter<bool>  ("use_lidar_distance",           true);
  desired_distance_m_           = declare_parameter<double>("desired_distance_m",           1.2);
  distance_kp_                  = declare_parameter<double>("distance_kp",                  0.6);
  distance_deadband_m_          = declare_parameter<double>("distance_deadband_m",          0.10);
  lidar_angle_window_scale_     = declare_parameter<double>("lidar_angle_window_scale",     1.0);
  lidar_min_window_rad_         = declare_parameter<double>("lidar_min_window_rad",         0.05);
  lidar_max_distance_m_         = declare_parameter<double>("lidar_max_distance_m",         8.0);
  camera_lidar_yaw_offset_rad_  = declare_parameter<double>("camera_lidar_yaw_offset_rad",  0.0);
  lidar_min_valid_points_       = declare_parameter<int>   ("lidar_min_valid_points",       3);
  distance_filter_window_       = declare_parameter<int>   ("distance_filter_window",       5);
  distance_ema_alpha_           = declare_parameter<double>("distance_ema_alpha",           0.4);

  // -- Bbox height fallback (secondary) --
  use_bbox_height_fallback_ = declare_parameter<bool>  ("use_bbox_height_fallback", true);
  linear_kp_                = declare_parameter<double>("linear_kp",               0.002);
  target_height_desired_    = declare_parameter<double>("target_height_desired",   300.0);
  bbox_height_deadband_     = declare_parameter<double>("bbox_height_deadband",    25.0);

  // -- Slot / target management --
  max_reacquire_dist_px_   = declare_parameter<double>("max_reacquire_dist_px",   300.0);
  manual_cmd_timeout_sec_  = declare_parameter<double>("manual_cmd_timeout_sec",  0.25);
  target_lost_timeout_sec_ = declare_parameter<double>("target_lost_timeout_sec", 0.60);
  slot_release_timeout_sec_= declare_parameter<double>("slot_release_timeout_sec",2.0);
  search_angular_z_        = declare_parameter<double>("search_angular_z",        0.40);
  control_rate_hz_         = declare_parameter<double>("control_rate_hz",         20.0);
}

// ---------------------------------------------------------------------------
// Interface creation
// ---------------------------------------------------------------------------

void TrackerNode::createInterfaces()
{
  tracking_sub_ = create_subscription<yolo_msgs::msg::DetectionArray>(
    tracking_topic_, rclcpp::SensorDataQoS(),
    std::bind(&TrackerNode::trackingCallback, this, std::placeholders::_1));

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    image_topic_, rclcpp::SensorDataQoS(),
    std::bind(&TrackerNode::imageCallback, this, std::placeholders::_1));

  scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
    scan_topic_, rclcpp::SensorDataQoS(),
    std::bind(&TrackerNode::scanCallback, this, std::placeholders::_1));

  manual_cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
    manual_cmd_topic_, 10,
    std::bind(&TrackerNode::manualCmdCallback, this, std::placeholders::_1));

  user_command_sub_ = create_subscription<std_msgs::msg::String>(
    user_command_topic_, 10,
    std::bind(&TrackerNode::userCommandCallback, this, std::placeholders::_1));

  cmd_vel_pub_        = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
  state_pub_          = create_publisher<std_msgs::msg::String>("/tracker/state", 10);
  locked_target_pub_  = create_publisher<std_msgs::msg::String>("/tracker/locked_target", 10);
  target_list_pub_    = create_publisher<std_msgs::msg::String>("/tracker/target_list", 10);
  overlay_image_pub_  = create_publisher<sensor_msgs::msg::Image>("/tracker/overlay_image", 10);
  target_distance_pub_= create_publisher<std_msgs::msg::Float64>("/tracker/target_distance", 10);

  const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, control_rate_hz_));
  control_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::milliseconds>(period),
    std::bind(&TrackerNode::controlLoop, this));
}

// ---------------------------------------------------------------------------
// Stable slot mapping
// ---------------------------------------------------------------------------

void TrackerNode::updateSlotMapping(std::vector<TrackedTarget> & targets)
{
  const rclcpp::Time current_time = now();

  for (const auto & t : targets) {
    id_last_seen_[t.tracking_id] = current_time;
  }

  std::vector<std::string> to_release;
  for (const auto & [id, last_seen] : id_last_seen_) {
    if ((current_time - last_seen).seconds() > slot_release_timeout_sec_) {
      to_release.push_back(id);
    }
  }
  for (const auto & id : to_release) {
    id_last_seen_.erase(id);
    const auto it = id_to_slot_.find(id);
    if (it != id_to_slot_.end()) {
      slot_to_id_.erase(it->second);
      id_to_slot_.erase(it);
    }
  }

  for (auto & t : targets) {
    if (id_to_slot_.count(t.tracking_id)) {
      t.slot = id_to_slot_.at(t.tracking_id);
    } else {
      for (int s = 1; s <= 9; ++s) {
        if (!slot_to_id_.count(s)) {
          slot_to_id_[s] = t.tracking_id;
          id_to_slot_[t.tracking_id] = s;
          t.slot = s;
          break;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void TrackerNode::trackingCallback(const yolo_msgs::msg::DetectionArray::SharedPtr msg)
{
  std::vector<TrackedTarget> new_targets;
  new_targets.reserve(msg->detections.size());

  const rclcpp::Time stamp =
    (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
    ? now() : rclcpp::Time(msg->header.stamp);

  int empty_id_counter = 0;
  for (const auto & det : msg->detections) {
    TrackedTarget target;
    target.tracking_id = det.id.empty()
      ? ("_anon_" + std::to_string(stamp.nanoseconds()) + "_" + std::to_string(empty_id_counter++))
      : det.id;
    target.class_id   = det.class_id;
    target.class_name = det.class_name;
    target.center_x   = det.bbox.center.position.x;
    target.center_y   = det.bbox.center.position.y;
    target.width      = det.bbox.size.x;
    target.height     = det.bbox.size.y;
    target.stamp      = stamp;
    new_targets.push_back(std::move(target));
  }

  updateSlotMapping(new_targets);
  latest_targets_ = std::move(new_targets);
  publishTargetList();
}

void TrackerNode::scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  latest_scan_ = msg;
}

void TrackerNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  latest_image_ = msg;
  if (overlay_image_pub_->get_subscription_count() == 0) return;

  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge: %s", e.what());
    return;
  }

  cv::Mat & frame = cv_ptr->image;
  const int font = cv::FONT_HERSHEY_SIMPLEX;

  // -- Per-target slot overlays --
  // Determine which slot the locked target occupies (if any)
  // Using slot rather than tracking_id for visual matching because ByteTrack
  // may reassign IDs frame-to-frame while the slot stays constant via our mapping.
  int locked_slot = -1;
  if (!locked_target_id_.empty()) {
    const auto it = id_to_slot_.find(locked_target_id_);
    if (it != id_to_slot_.end()) locked_slot = it->second;
  }

  for (const auto & t : latest_targets_) {
    if (t.slot < 1) continue;

    const int x1 = static_cast<int>(t.center_x - t.width  / 2.0);
    const int y1 = static_cast<int>(t.center_y - t.height / 2.0);
    const int x2 = static_cast<int>(t.center_x + t.width  / 2.0);
    const int y2 = static_cast<int>(t.center_y + t.height / 2.0);

    const bool is_locked   = (t.slot == locked_slot);
    const cv::Scalar color = is_locked ? cv::Scalar(0, 50, 255) : cv::Scalar(0, 200, 0);
    const int thickness    = is_locked ? 3 : 2;

    cv::rectangle(frame, {x1, y1}, {x2, y2}, color, thickness);

    const std::string slot_label = std::to_string(t.slot);
    const cv::Point label_org{std::max(x1, 4), std::max(y1 - 8, 20)};
    cv::putText(frame, slot_label, label_org + cv::Point(2, 2), font, 1.4,
      cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(frame, slot_label, label_org, font, 1.4,
      is_locked ? cv::Scalar(0, 50, 255) : cv::Scalar(50, 255, 50), 3, cv::LINE_AA);

    const std::string info = t.class_name + (is_locked ? " [LOCK]" : "");
    cv::putText(frame, info, {x1, y2 + 18}, font, 0.55, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(frame, info, {x1, y2 + 18}, font, 0.55, color, 1, cv::LINE_AA);
  }

  // -- Status overlay (top-left, three lines) --
  // Line 1: FSM state + locked id
  const std::string line1 = "State: " + stateToString(state_) +
    (locked_target_id_.empty() ? "" : "  id=" + locked_target_id_);

  // Line 2: LiDAR distance (or fallback tag)
  std::string line2;
  if (state_ == TrackerState::Locked || state_ == TrackerState::Searching) {
    if (last_measured_distance_m_ > 0.0) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << last_measured_distance_m_;
      line2 = "Dist(LiDAR): " + oss.str() + " m";
    } else if (use_lidar_distance_) {
      line2 = use_bbox_height_fallback_ ? "Dist: bbox fallback" : "Dist: no reading";
    }
  }

  // Draw with black shadow for readability on any background
  auto draw_text_line = [&](const std::string & text, int row) {
    const cv::Point pos{10, 30 + row * 28};
    cv::putText(frame, text, pos + cv::Point(2, 2), font, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(frame, text, pos, font, 0.7, cv::Scalar(255, 255, 100), 1, cv::LINE_AA);
  };
  draw_text_line(line1, 0);
  if (!line2.empty()) draw_text_line(line2, 1);

  overlay_image_pub_->publish(*cv_ptr->toImageMsg());
}

void TrackerNode::manualCmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  latest_manual_cmd_ = *msg;
  latest_manual_cmd_time_ = now();
}

void TrackerNode::userCommandCallback(const std_msgs::msg::String::SharedPtr msg)
{
  const std::string command = trim(msg->data);
  if (command.empty()) return;

  // Direct lock by real tracking ID: "lock:1364"
  if (command.rfind("lock:", 0) == 0) {
    const auto id = trim(command.substr(5));
    if (id.empty()) { RCLCPP_WARN(get_logger(), "lock: empty id"); return; }
    locked_target_id_  = id;
    state_             = TrackerState::Locked;
    last_target_seen_time_ = now();
    filter_initialized_ = false;
    forward_allowed_    = false;
    last_measured_distance_m_ = -1.0;
    dist_filter_initialized_ = false;
    distance_history_.clear();
    for (const auto & t : latest_targets_) {
      if (t.tracking_id == id) { locked_class_name_ = t.class_name; break; }
    }
    RCLCPP_INFO(get_logger(), "Locked -> id=%s (%s)", id.c_str(), locked_class_name_.c_str());
    return;
  }

  // Lock by stable slot: "slot:2"
  if (command.rfind("slot:", 0) == 0) {
    try {
      const int slot = std::stoi(trim(command.substr(5)));
      const auto it  = slot_to_id_.find(slot);
      if (it == slot_to_id_.end()) {
        RCLCPP_WARN(get_logger(), "Slot %d has no target.", slot); return;
      }
      locked_target_id_  = it->second;
      state_             = TrackerState::Locked;
      last_target_seen_time_ = now();
      filter_initialized_ = false;
      forward_allowed_    = false;
      last_measured_distance_m_ = -1.0;
      dist_filter_initialized_ = false;
      distance_history_.clear();
      locked_class_name_.clear();
      for (const auto & t : latest_targets_) {
        if (t.tracking_id == locked_target_id_) { locked_class_name_ = t.class_name; break; }
      }
      RCLCPP_INFO(get_logger(), "Slot %d -> Locked id=%s (%s)",
        slot, locked_target_id_.c_str(), locked_class_name_.c_str());
    } catch (const std::exception &) {
      RCLCPP_WARN(get_logger(), "Invalid slot value in: '%s'", command.c_str());
    }
    return;
  }

  if (command == "unlock" || command == "manual") {
    state_ = TrackerState::Manual;
    locked_target_id_.clear();
    last_measured_distance_m_ = -1.0;
    RCLCPP_INFO(get_logger(), "Manual state.");
    return;
  }

  if (command == "search") {
    if (locked_target_id_.empty()) {
      RCLCPP_WARN(get_logger(), "search: no target locked."); return;
    }
    state_ = TrackerState::Searching;
    RCLCPP_INFO(get_logger(), "Searching for id=%s", locked_target_id_.c_str());
    return;
  }

  RCLCPP_WARN(get_logger(), "Unknown command: '%s'", command.c_str());
}

// ---------------------------------------------------------------------------
// Control loop
// ---------------------------------------------------------------------------

void TrackerNode::controlLoop()
{
  const auto current_time = now();
  geometry_msgs::msg::Twist cmd = zeroTwist();

  switch (state_) {
    case TrackerState::Manual: {
      if ((current_time - latest_manual_cmd_time_).seconds() <= manual_cmd_timeout_sec_) {
        cmd = latest_manual_cmd_;
      }
      break;
    }

    case TrackerState::Locked: {
      if (locked_target_id_.empty()) { state_ = TrackerState::Manual; break; }
      auto target = findTargetById(locked_target_id_);

      // If exact ID not found, immediately try same-class fallback
      // (don't wait for target_lost_timeout — ByteTrack often reassigns IDs
      //  on minor occlusions, we want seamless handoff without a control gap)
      if (!target.has_value()) {
        auto fallback = findNearestSameClass();
        if (fallback.has_value()) {
          const std::string old_id = locked_target_id_;
          locked_target_id_ = fallback->tracking_id;
          // Transfer slot mapping from old ID to new ID
          const auto slot_it = id_to_slot_.find(old_id);
          if (slot_it != id_to_slot_.end()) {
            const int slot = slot_it->second;
            slot_to_id_[slot] = locked_target_id_;
            id_to_slot_.erase(old_id);
            id_to_slot_[locked_target_id_] = slot;
          }
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "ID handoff: %s -> %s (%s)", old_id.c_str(),
            locked_target_id_.c_str(), locked_class_name_.c_str());
          target = fallback;
        }
      }

      if (target.has_value()) {
        last_target_seen_time_ = current_time;
        last_known_center_x_   = target->center_x;
        last_known_center_y_   = target->center_y;
        locked_class_name_     = target->class_name;
        cmd = computeTrackingCommand(*target);
      } else if ((current_time - last_target_seen_time_).seconds() > target_lost_timeout_sec_) {
        state_ = TrackerState::Searching;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
          "Target id=%s lost -> Searching.", locked_target_id_.c_str());
      }
      break;
    }

    case TrackerState::Searching: {
      if (locked_target_id_.empty()) { state_ = TrackerState::Manual; break; }

      auto target = findTargetById(locked_target_id_);

      if (!target.has_value()) {
        target = findNearestSameClass();
        if (target.has_value()) {
          const std::string old_id = locked_target_id_;
          locked_target_id_ = target->tracking_id;
          const auto slot_it = id_to_slot_.find(old_id);
          if (slot_it != id_to_slot_.end()) {
            const int slot = slot_it->second;
            slot_to_id_[slot] = locked_target_id_;
            id_to_slot_.erase(old_id);
            id_to_slot_[locked_target_id_] = slot;
          }
          RCLCPP_INFO(get_logger(), "Same-class fallback: %s -> %s (%s)",
            old_id.c_str(), locked_target_id_.c_str(), locked_class_name_.c_str());
        }
      }

      if (target.has_value()) {
        state_ = TrackerState::Locked;
        last_target_seen_time_ = current_time;
        filter_initialized_ = false;
        forward_allowed_ = false;
        dist_filter_initialized_ = false;
        distance_history_.clear();
        cmd = computeTrackingCommand(*target);
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
          "Reacquired id=%s -> Locked.", locked_target_id_.c_str());
      } else {
        last_measured_distance_m_ = -1.0;
        cmd = computeSearchCommand();
      }
      break;
    }
  }

  cmd_vel_pub_->publish(cmd);

  std_msgs::msg::String state_msg;
  state_msg.data = stateToString(state_);
  state_pub_->publish(state_msg);

  std_msgs::msg::String locked_msg;
  locked_msg.data = locked_target_id_;
  locked_target_pub_->publish(locked_msg);

  std_msgs::msg::Float64 dist_msg;
  dist_msg.data = last_measured_distance_m_;
  target_distance_pub_->publish(dist_msg);
}

// ---------------------------------------------------------------------------
// Helpers — target lookup
// ---------------------------------------------------------------------------

std::optional<TrackedTarget> TrackerNode::findTargetById(const std::string & tracking_id) const
{
  const auto it = std::find_if(
    latest_targets_.begin(), latest_targets_.end(),
    [&tracking_id](const TrackedTarget & t) { return t.tracking_id == tracking_id; });
  return (it == latest_targets_.end()) ? std::nullopt : std::make_optional(*it);
}

std::optional<TrackedTarget> TrackerNode::findNearestSameClass() const
{
  if (locked_class_name_.empty()) return std::nullopt;

  const TrackedTarget * best = nullptr;
  double best_dist2 = max_reacquire_dist_px_ * max_reacquire_dist_px_;

  for (const auto & t : latest_targets_) {
    if (t.class_name != locked_class_name_) continue;
    const double dx = t.center_x - last_known_center_x_;
    const double dy = t.center_y - last_known_center_y_;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_dist2) { best_dist2 = d2; best = &t; }
  }
  return best ? std::make_optional(*best) : std::nullopt;
}

// ---------------------------------------------------------------------------
// Helpers — LiDAR ROI distance extraction
// ---------------------------------------------------------------------------

// Normalize any angle to [0, 2π) so it can be compared directly against
// scan frames where angle_min = 0 and angle_max ≈ 2π.
double TrackerNode::normalizeAngle2Pi(double a)
{
  constexpr double TWO_PI = 2.0 * M_PI;
  a = std::fmod(a, TWO_PI);
  if (a < 0.0) a += TWO_PI;
  return a;
}

// extractRoiDistance
// -----------------
// angle_center_rad : camera bearing of target in sensor frame (0 = forward, positive = CCW/left)
// angle_half_span_rad : half-width of the sampling window (rad)
//
// Scan frame: angle_min=0, angle_max≈2π, increases CCW.
// Camera bearing → LiDAR angle (after yaw-offset correction) may be negative or > 2π,
// so we normalize before indexing.
//
// Wrap-around: if the window crosses 0/2π we split into two index ranges and merge.
//
// Spatial aggregation: collect all valid points, sort, take mean of front-k (smallest k),
// where k = min(3, valid.size()). Robust against background clutter while more
// stable than a single percentile sample.
//
// Minimum valid points guard: if fewer than lidar_min_valid_points_ survive filtering
// we return nullopt (don't trust a reading with almost no data).

std::optional<double> TrackerNode::extractRoiDistance(
  const sensor_msgs::msg::LaserScan & scan,
  double angle_center_rad,
  double angle_half_span_rad) const
{
  if (scan.angle_increment <= 0.0 || scan.ranges.empty()) return std::nullopt;

  const int n = static_cast<int>(scan.ranges.size());

  // Normalize the window bounds into [0, 2π)
  const double a_lo_raw = angle_center_rad - angle_half_span_rad;
  const double a_hi_raw = angle_center_rad + angle_half_span_rad;
  const double a_lo = normalizeAngle2Pi(a_lo_raw);
  const double a_hi = normalizeAngle2Pi(a_hi_raw);

  // Convert a normalized angle to the nearest scan index [0, n-1]
  auto angle_to_idx = [&](double a) -> int {
    int idx = static_cast<int>((a - scan.angle_min) / scan.angle_increment);
    return std::clamp(idx, 0, n - 1);
  };

  // Collect valid ranges, handling wrap-around across 0/2π
  std::vector<double> valid;
  valid.reserve(static_cast<size_t>(2.0 * angle_half_span_rad / scan.angle_increment + 2));

  auto collect = [&](int lo_idx, int hi_idx) {
    if (lo_idx > hi_idx) std::swap(lo_idx, hi_idx);
    for (int i = lo_idx; i <= hi_idx; ++i) {
      const float r = scan.ranges[i];
      if (!std::isfinite(r)) continue;
      if (r < scan.range_min || r > scan.range_max) continue;
      if (static_cast<double>(r) > lidar_max_distance_m_) continue;
      valid.push_back(static_cast<double>(r));
    }
  };

  const bool wraps = (a_lo > a_hi);  // window crosses 0/2π boundary
  if (wraps) {
    // Two segments: [a_lo, 2π) and [0, a_hi]
    collect(angle_to_idx(a_lo), n - 1);
    collect(0, angle_to_idx(a_hi));
  } else {
    collect(angle_to_idx(a_lo), angle_to_idx(a_hi));
  }

  if (static_cast<int>(valid.size()) < lidar_min_valid_points_) return std::nullopt;

  // Front-k mean: average the k smallest values (k = min(3, size))
  std::sort(valid.begin(), valid.end());
  const size_t k = std::min<size_t>(3, valid.size());
  double sum = 0.0;
  for (size_t i = 0; i < k; ++i) sum += valid[i];
  return sum / static_cast<double>(k);
}

// Per-class window scale overrides (priority 3)
double TrackerNode::getClassWindowScale(const std::string & class_name) const
{
  if (class_name == "person")        return lidar_angle_window_scale_ * 1.4;
  if (class_name == "fire hydrant")  return lidar_angle_window_scale_ * 0.8;
  return lidar_angle_window_scale_;
}

double TrackerNode::getClassMinWindowRad(const std::string & class_name) const
{
  if (class_name == "fire hydrant")  return std::max(lidar_min_window_rad_, 0.08);
  return lidar_min_window_rad_;
}

// getDistanceForTarget
// --------------------
// Maps the current filtered pixel position to a LiDAR bearing.
//
// Sign convention:
//   pixel_offset = filtered_center_x_ - image_center_x_
//   > 0 : target is to the RIGHT of image center (camera looks right -> bearing < 0 / CW)
//   < 0 : target is to the LEFT  of image center (camera looks left  -> bearing > 0 / CCW)
//
// In a standard forward-facing camera:
//   image x increases LEFT→RIGHT
//   LiDAR angle increases CCW (left when viewed from above)
//   So: right-of-center (pixel_offset > 0) = negative bearing (CW = "right")
//       left-of-center  (pixel_offset < 0) = positive bearing (CCW = "left")
// → angle_center = -(pixel_offset / image_width_) * horizontal_fov_rad_
//
// Then add camera_lidar_yaw_offset_rad_ for physical mounting offset,
// and normalise to [0, 2π) before passing to extractRoiDistance.

std::optional<double> TrackerNode::getDistanceForTarget(const TrackedTarget & target) const
{
  if (!latest_scan_) return std::nullopt;

  // Negative sign: rightward pixel offset → negative (clockwise) bearing
  const double pixel_offset  = filtered_center_x_ - image_center_x_;
  const double camera_bearing = -(pixel_offset / image_width_) * horizontal_fov_rad_;
  const double angle_center   = normalizeAngle2Pi(camera_bearing + camera_lidar_yaw_offset_rad_);

  // Per-class angular window
  const double window_scale  = getClassWindowScale(target.class_name);
  const double min_window    = getClassMinWindowRad(target.class_name);
  const double bbox_angle_w  = (target.width / image_width_) * horizontal_fov_rad_;
  const double angle_half_span = std::max(0.5 * bbox_angle_w * window_scale, min_window);

  RCLCPP_DEBUG(get_logger(),
    "LiDAR ROI: pixel_off=%.1f bearing=%.3f rad (%.1f deg) half_span=%.3f rad",
    pixel_offset, angle_center, angle_center * 180.0 / M_PI, angle_half_span);

  return extractRoiDistance(*latest_scan_, angle_center, angle_half_span);
}

// updateDistanceFilter
// --------------------
// Spatial reading -> sliding-window median/percentile -> EMA -> filtered output.
// Reset state by calling with a negative value or reassigning dist_filter_initialized_.

double TrackerNode::updateDistanceFilter(double spatial_reading)
{
  // Maintain sliding window of fixed length
  distance_history_.push_back(spatial_reading);
  while (static_cast<int>(distance_history_.size()) > distance_filter_window_) {
    distance_history_.pop_front();
  }

  // Window median (robust against single outlier frames)
  std::vector<double> sorted_win(distance_history_.begin(), distance_history_.end());
  std::sort(sorted_win.begin(), sorted_win.end());
  const double window_median = sorted_win[sorted_win.size() / 2];

  // EMA on top of the median
  if (!dist_filter_initialized_) {
    filtered_distance_m_ = window_median;
    dist_filter_initialized_ = true;
  } else {
    filtered_distance_m_ =
      distance_ema_alpha_ * window_median + (1.0 - distance_ema_alpha_) * filtered_distance_m_;
  }
  return filtered_distance_m_;
}

// ---------------------------------------------------------------------------
// Control command computation
// ---------------------------------------------------------------------------

geometry_msgs::msg::Twist TrackerNode::computeTrackingCommand(const TrackedTarget & target)
{
  geometry_msgs::msg::Twist cmd = zeroTwist();

  // ---- 1. EMA filter on raw center_x ----
  if (!filter_initialized_) {
    filtered_center_x_ = target.center_x;
    filter_initialized_ = true;
  } else {
    filtered_center_x_ =
      center_x_alpha_ * target.center_x + (1.0 - center_x_alpha_) * filtered_center_x_;
  }

  // ---- 2. Angular (yaw) control with deadband ----
  // error_x > 0: target is LEFT  of center -> turn left  -> angular_z > 0
  // error_x < 0: target is RIGHT of center -> turn right -> angular_z < 0
  const double error_x = image_center_x_ - filtered_center_x_;

  double raw_angular_z = 0.0;
  if (std::abs(error_x) > yaw_deadband_px_) {
    const double error_beyond_db =
      (error_x > 0) ? (error_x - yaw_deadband_px_) : (error_x + yaw_deadband_px_);
    raw_angular_z = std::clamp(yaw_kp_ * error_beyond_db, -max_angular_z_, max_angular_z_);
  }

  double angular_z = 0.0;
  if (std::abs(error_x) <= yaw_deadband_px_) {
    prev_angular_z_ = 0.0;  // snap to zero inside deadband
  } else {
    angular_z = angular_z_alpha_ * raw_angular_z + (1.0 - angular_z_alpha_) * prev_angular_z_;
    prev_angular_z_ = angular_z;
  }

  // ---- 3. Hysteresis gate for forward motion ----
  // 仅控制"前进"，不影响 LiDAR 判断的后退（太近时需要及时倒退）
  const double abs_err = std::abs(error_x);
  if (!forward_allowed_ && abs_err <= forward_gate_open_px_) {
    forward_allowed_ = true;
  } else if (forward_allowed_ && abs_err > forward_gate_close_px_) {
    forward_allowed_ = false;
  }

  // ---- 4. Linear (distance) control ----
  double linear_x = 0.0;
  bool lidar_used = false;

  // --- Primary: LiDAR ROI distance ---
  // LiDAR 后退（太近）不受 forward_allowed_ 限制，确保安全距离优先
  if (use_lidar_distance_) {
    const auto dist = getDistanceForTarget(target);
    if (dist.has_value()) {
      const double smoothed = updateDistanceFilter(*dist);
      last_measured_distance_m_ = smoothed;
      lidar_used = true;

      // distance_error > 0: measured > desired -> too far  -> move forward (+)
      // distance_error < 0: measured < desired -> too close-> move backward (-)
      const double distance_error = smoothed - desired_distance_m_;
      if (std::abs(distance_error) > distance_deadband_m_) {
        const double raw = distance_kp_ * distance_error;
        if (raw > 0.0) {
          // 前进：受 forward_allowed_ 门控
          linear_x = forward_allowed_ ? raw : 0.0;
        } else {
          // 后退：绕过 forward_allowed_，允许及时后退
          linear_x = allow_reverse_ ? raw : 0.0;
        }
      }
    } else {
      last_measured_distance_m_ = -1.0;
      distance_history_.clear();
      dist_filter_initialized_ = false;
    }
  }

  // --- Secondary: bbox height fallback (仅 forward_allowed_ 时生效) ---
  if (!lidar_used && use_bbox_height_fallback_ && forward_allowed_) {
    const double height_error = target_height_desired_ - target.height;
    if (height_error > bbox_height_deadband_) {
      linear_x = linear_kp_ * height_error;
    } else if (allow_reverse_ && height_error < -bbox_height_deadband_) {
      linear_x = linear_kp_ * height_error;
    }
  }

  // --- Clamp ---
  linear_x = std::clamp(linear_x, -max_reverse_x_, max_linear_x_);

  cmd.angular.z = angular_z;
  cmd.linear.x  = linear_x;
  return cmd;
}

geometry_msgs::msg::Twist TrackerNode::computeSearchCommand() const
{
  geometry_msgs::msg::Twist cmd = zeroTwist();
  cmd.angular.z = search_angular_z_;
  return cmd;
}

geometry_msgs::msg::Twist TrackerNode::zeroTwist() const
{
  return geometry_msgs::msg::Twist();
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

std::string TrackerNode::trim(const std::string & input)
{
  const auto begin = input.find_first_not_of(" 	
");
  if (begin == std::string::npos) return {};
  const auto end = input.find_last_not_of(" 	
");
  return input.substr(begin, end - begin + 1);
}

std::string TrackerNode::stateToString(TrackerState state)
{
  switch (state) {
    case TrackerState::Manual:    return "Manual";
    case TrackerState::Locked:    return "Locked";
    case TrackerState::Searching: return "Searching";
    default:                      return "Unknown";
  }
}

void TrackerNode::publishTargetList()
{
  std_msgs::msg::String msg;
  msg.data = buildTargetListJson();
  target_list_pub_->publish(msg);
}

std::string TrackerNode::buildTargetListJson() const
{
  std::vector<const TrackedTarget *> sorted;
  for (const auto & t : latest_targets_) {
    if (t.slot >= 1) sorted.push_back(&t);
  }
  std::sort(sorted.begin(), sorted.end(),
    [](const TrackedTarget * a, const TrackedTarget * b) { return a->slot < b->slot; });

  // Determine locked slot for stable matching
  int locked_slot = -1;
  if (!locked_target_id_.empty()) {
    const auto it = id_to_slot_.find(locked_target_id_);
    if (it != id_to_slot_.end()) locked_slot = it->second;
  }

  std::ostringstream oss;
  oss << "[";
  bool first = true;
  for (const auto * t : sorted) {
    if (!first) oss << ",";
    first = false;
    const bool is_locked = (t->slot == locked_slot);
    oss << "{\"slot\":"   << t->slot
        << ",\"id\":\""   << t->tracking_id << "\""
        << ",\"class\":\"" << t->class_name << "\""
        << ",\"h\":"      << static_cast<int>(t->height)
        << ",\"locked\":" << (is_locked ? "true" : "false") << "}";
  }
  oss << "]";
  return oss.str();
}

}  // namespace interactive_tracker_cpp
