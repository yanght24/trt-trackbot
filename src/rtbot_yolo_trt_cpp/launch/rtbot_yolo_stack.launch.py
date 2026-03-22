"""
rtbot_yolo_stack.launch.py — v2 全 C++ 替代方案

完全替代 Python yolo_ros（yolo_node.py + tracking_node.py + debug_node.py）。
三个节点全部为 C++ 实现，不再依赖 yolo_ros 包。

═══════════════════════════════════════════════════════════════════════
第二版终端 3 新命令：
  source /path/to/trt-trackbot/scripts/env_setup.sh && ros2 launch rtbot_yolo_trt_cpp rtbot_yolo_stack.launch.py \
    engine_path:=/path/to/models/yolo11n_e2e_fp16.engine \
    input_image_topic:=/camera/image_raw \
    threshold:=0.3 \
    use_sim_time:=True

终端 1/2/4/5/6/7 保持不变。
═══════════════════════════════════════════════════════════════════════
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share = get_package_share_directory("rtbot_yolo_trt_cpp")
    default_yaml = os.path.join(pkg_share, "config", "stack.yaml")

    # ── Launch arguments ──────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("engine_path",
            description="Path to end2end TensorRT .engine file (required)"),
        DeclareLaunchArgument("input_image_topic",
            default_value="/camera/image_raw"),
        DeclareLaunchArgument("threshold", default_value="0.3",
            description="Confidence threshold"),
        DeclareLaunchArgument("namespace", default_value="yolo"),
        DeclareLaunchArgument("use_sim_time", default_value="False"),
    ]

    ns = LaunchConfiguration("namespace")
    engine_path = LaunchConfiguration("engine_path")
    input_image_topic = LaunchConfiguration("input_image_topic")
    threshold = LaunchConfiguration("threshold")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # ── 1. C++ TensorRT Detector (end2end) ────────────────────────────
    detector_node = Node(
        package="rtbot_yolo_trt_cpp",
        executable="detector_node",
        name="detector_node",
        namespace=ns,
        parameters=[
            default_yaml,
            {
                "engine_path": engine_path,
                "conf_threshold": threshold,
                "detection_topic": "detections",
                "use_sim_time": use_sim_time,
            },
        ],
        remappings=[("image_raw", input_image_topic)],
        output="screen",
    )

    # ── 2. C++ Tracker ────────────────────────────────────────────────
    tracker_node = Node(
        package="rtbot_yolo_trt_cpp",
        executable="tracker_node",
        name="tracker_node",
        namespace=ns,
        parameters=[
            default_yaml,
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )

    # ── 3. C++ Debug Overlay ──────────────────────────────────────────
    debug_node = Node(
        package="rtbot_yolo_trt_cpp",
        executable="debug_node",
        name="debug_node",
        namespace=ns,
        parameters=[
            {"use_sim_time": use_sim_time},
        ],
        remappings=[("image_raw", input_image_topic)],
        output="screen",
    )

    return LaunchDescription(args + [detector_node, tracker_node, debug_node])
