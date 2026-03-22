"""
rtbot_yolo_trt.launch.py — V1 混合方案

C++ TensorRT detector (raw-head [1,84,N] + CPU NMS) + Python tracking_node + Python debug_node.
使用标准 YOLO engine（非 end2end），替换 Python yolo_node 为 C++ detector_node_v1。

用法:
  source /path/to/trt-trackbot/scripts/env_setup.sh && ros2 launch rtbot_yolo_trt_cpp rtbot_yolo_trt.launch.py \
    engine_path:=/path/to/models/yolo11n_fp16.engine \
    input_image_topic:=/camera/image_raw \
    threshold:=0.3 \
    iou:=0.7 \
    use_sim_time:=True
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # -- Launch arguments --------------------------------------------------
    args = [
        DeclareLaunchArgument("engine_path",
            description="Path to standard (raw-head) TensorRT .engine file (required)"),
        DeclareLaunchArgument("input_image_topic",
            default_value="/camera/image_raw"),
        DeclareLaunchArgument("threshold", default_value="0.3",
            description="Confidence threshold"),
        DeclareLaunchArgument("iou", default_value="0.7",
            description="NMS IoU threshold"),
        DeclareLaunchArgument("max_det", default_value="300"),
        DeclareLaunchArgument("tracker", default_value="bytetrack.yaml"),
        DeclareLaunchArgument("namespace", default_value="yolo"),
        DeclareLaunchArgument("use_sim_time", default_value="False"),
    ]

    ns = LaunchConfiguration("namespace")
    engine_path = LaunchConfiguration("engine_path")
    input_image_topic = LaunchConfiguration("input_image_topic")
    threshold = LaunchConfiguration("threshold")
    iou = LaunchConfiguration("iou")
    max_det = LaunchConfiguration("max_det")
    tracker = LaunchConfiguration("tracker")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # -- 1. C++ TensorRT Detector V1 (raw-head + CPU NMS) -----------------
    detector_node = Node(
        package="rtbot_yolo_trt_cpp",
        executable="detector_node_v1",
        name="yolo_node",
        namespace=ns,
        parameters=[{
            "engine_path": engine_path,
            "conf_threshold": threshold,
            "iou_threshold": iou,
            "max_det": max_det,
            "detection_topic": "detections",
            "use_sim_time": use_sim_time,
        }],
        remappings=[("image_raw", input_image_topic)],
        output="screen",
    )

    # -- 2. Existing Python tracking_node ---------------------------------
    tracking_node = Node(
        package="yolo_ros",
        executable="tracking_node",
        name="tracking_node",
        namespace=ns,
        parameters=[{
            "tracker": tracker,
            "image_reliability": 2,
            "use_sim_time": use_sim_time,
        }],
        remappings=[("image_raw", input_image_topic)],
        output="screen",
    )

    # -- 3. Existing Python debug_node ------------------------------------
    debug_node = Node(
        package="yolo_ros",
        executable="debug_node",
        name="debug_node",
        namespace=ns,
        parameters=[{
            "image_reliability": 2,
            "use_sim_time": use_sim_time,
        }],
        remappings=[
            ("image_raw", input_image_topic),
            ("detections", "tracking"),
        ],
        output="screen",
    )

    return LaunchDescription(args + [detector_node, tracking_node, debug_node])
