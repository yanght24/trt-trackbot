# trt-trackbot

**ROS2 + TensorRT YOLO multi-modal object tracking bot** — end-to-end GPU pipeline from raw camera image to `/cmd_vel` motor command, running on TurtleBot3 Waffle (simulation).

> **trt-trackbot** = TensorRT detector (v2 end2end, CPU-NMS-free) + ByteTrack + C++ FSM controller + LiDAR distance control

---

## 中文说明 (Chinese)

ROS2 Humble + Gazebo 仿真下的小车多模态目标跟踪系统。

- 全流程 GPU 推理：图像 → TensorRT end2end YOLO → ByteTrack 追踪 → C++ FSM 控制器 → `/cmd_vel`
- LiDAR 距离闭环：订阅 `/scan`，20th 百分位距离估计，前向速度 PD 控制
- 交互式目标锁定：键盘按 `1-9` 锁定槽位目标，支持同类别重识别 fallback
- 稳定槽位映射：ByteTrack 大 ID（如 `1364`）自动映射到 `1-9` 显示槽位

---

## Features

| Feature | Detail |
|---------|--------|
| Detector | YOLOv11n FP16 TensorRT end2end (EfficientNMS_TRT embedded) |
| Tracker | ByteTrack via `yolo_ros` |
| Controller | C++ FSM: Manual / Locked / Searching |
| Distance | LiDAR `/scan` → 20th-percentile ROI distance |
| Slot mapping | ByteTrack large IDs → keyboard-friendly slots 1–9 |
| Overlay | Real-time annotated image on `/tracker/overlay_image` |
| Benchmark | Python TRT 29ms → C++ TRT v1 8.6ms → **C++ TRT v2 2.4ms** |

---

## Performance

All benchmarks on **NVIDIA RTX 4070 Laptop GPU**, resolution **1920×1080**, 30-second steady-state recording.

| Backend | Mean FPS | Mean Latency | GPU Power | GPU Util |
|---------|----------|-------------|-----------|----------|
| Python TRT (baseline) | 27.8 | 29.0 ms | 43.5 W | 41% |
| C++ TRT v1 raw-head | 29.5 | 8.6 ms | 30.1 W | 31% |
| **C++ TRT v2 end2end** | **29.7** | **2.4 ms** | **25.8 W** | **26%** |

> v2 end2end removes CPU-side NMS entirely — all postprocessing runs inside TensorRT via `EfficientNMS_TRT`.
> Latency improvement: **12× vs Python**, **3.6× vs C++ v1**.

Generate charts:
```bash
python3 docs/benchmark_chart.py
# → docs/benchmark_fps.png, docs/benchmark_latency.png
```

---

## Architecture

```
/camera/image_raw
        │
        ▼
┌─────────────────────────────────────┐
│  rtbot_yolo_trt_cpp                 │
│  DetectorNode (TRT end2end, FP16)   │  GPU: letterbox → infer → NMS (TRT)
│    → /yolo/detections               │
│  TrackerNode (ByteTrack)            │
│    → /yolo/tracking                 │
└─────────────────────────────────────┘
        │                     │
        │               /scan (LiDAR)
        ▼                     │
┌─────────────────────────────────────┐
│  interactive_tracker_cpp            │
│  TrackerManagerNode                 │
│    FSM: Manual ↔ Locked ↔ Searching │
│    Slot mapping: ByteTrack ID→1-9   │
│    LiDAR distance PD control        │
│    EMA yaw filter + deadband        │
│    → /cmd_vel                       │
│    → /tracker/overlay_image         │
│    → /tracker/target_list (JSON)    │
│    → /tracker/target_distance       │
└─────────────────────────────────────┘
        │
        ▼
  TurtleBot3 Waffle
```

---

## Prerequisites

| Component | Version |
|-----------|--------|
| Ubuntu | 22.04 |
| ROS2 | Humble |
| Gazebo | Classic (gazebo11) |
| CUDA | 12.x |
| TensorRT | 8.6+ |
| OpenCV | 4.x |
| Python | 3.10 |

### Install ROS2 Humble

```bash
# Standard ROS2 Humble install
sudo apt install ros-humble-desktop ros-humble-cv-bridge ros-humble-image-transport
```

### Install TurtleBot3

```bash
sudo apt install ros-humble-turtlebot3 ros-humble-turtlebot3-simulations
export TURTLEBOT3_MODEL=waffle
```

### Clone yolo_ros (ByteTrack + yolo_msgs)

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/mgonzs13/yolo_ros.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select yolo_msgs
source install/setup.bash
```

### Export YOLO TensorRT Engine (end2end)

This project requires a **v2 end2end engine** with `EfficientNMS_TRT` embedded:

```bash
# Install ultralytics
pip install ultralytics

# Export YOLO11n to ONNX with end2end NMS
# Use TensorRT-For-YOLO-Series or ultralytics export:
python3 -c "
from ultralytics import YOLO
m = YOLO('yolo11n.pt')
m.export(format='engine', half=True, imgsz=640, device=0,
         simplify=True, nms=True)
"
# Or use the provided export script:
bash src/rtbot_yolo_trt_cpp/scripts/export_e2e.sh yolo11n.pt /path/to/output.engine
```

> **Note:** The `.engine` file is hardware-specific and not included in this repo.
> Rebuild on your target machine.

---

## Build

```bash
# 1. Clone this repo into a ROS2 workspace
mkdir -p ~/trt_ws/src
cd ~/trt_ws/src
git clone https://github.com/yanght24/trt-trackbot.git .

# 2. Source ROS2 and yolo_msgs
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash  # yolo_msgs

# 3. Build
cd ~/trt_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Quick Start (Simulation)

Open **5 terminals**, each with:
```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle
source ~/trt_ws/install/setup.bash
```

### Terminal 1 — Gazebo
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py use_sim_time:=True
```

### Terminal 2 — Dynamic Patrol (optional)
```bash
python3 sim_assets/scripts/patrol_entities.py
```

### Terminal 3 — TRT Detector + Tracker (V2 full stack)
```bash
ros2 launch rtbot_yolo_trt_cpp rtbot_yolo_stack.launch.py \
  engine_path:=/path/to/yolo11n_fp16.engine \
  input_image_topic:=/camera/image_raw \
  threshold:=0.3 \
  use_sim_time:=True
```

### Terminal 4 — FSM Controller
```bash
ros2 launch interactive_tracker_cpp tracker_system.launch.py
```

### Terminal 5 — Keyboard Control
```bash
ros2 run interactive_tracker_cpp keyboard_command_node.py
```

**Keyboard bindings:**

| Key | Action |
|-----|--------|
| `1`–`9` | Lock target in slot N |
| `u` | Unlock / return to Manual |
| `s` | Start Searching |
| `w/a/s/d` | Manual velocity |
| `q` | Quit |

---

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/yolo/tracking` | `yolo_msgs/DetectionArray` | ByteTrack detections with stable IDs |
| `/user_command` | `std_msgs/String` | `lock:ID` / `slot:N` / `unlock` / `search` |
| `/manual_cmd_vel` | `geometry_msgs/Twist` | Keyboard manual velocity |
| `/cmd_vel` | `geometry_msgs/Twist` | Output motor command |
| `/tracker/target_list` | `std_msgs/String` | JSON slot→ID map + locked flag |
| `/tracker/overlay_image` | `sensor_msgs/Image` | Annotated camera image |
| `/tracker/target_distance` | `std_msgs/Float64` | Current LiDAR distance to target (m) |

---

## Configuration

See `src/interactive_tracker_cpp/config/tracker_params.yaml`:

```yaml
tracker_manager_node:
  ros__parameters:
    target_distance_m: 1.2      # Desired following distance (m)
    linear_kp: 0.5              # PD proportional gain
    angular_kp: 1.2             # Yaw P gain
    ema_alpha: 0.3              # EMA smoothing for yaw
    deadband_px: 15             # Yaw deadband (pixels)
    slot_timeout_sec: 3.0       # Slot release delay after target disappears
```

---

## Project Structure

```
trt-trackbot/
├── src/
│   ├── rtbot_yolo_trt_cpp/         # TRT detector + ByteTrack ROS2 node (C++)
│   │   ├── include/                # Headers: TrtBackend, DetectorNode, TrackerNode
│   │   ├── src/                    # detector_node.cpp, tracker_node.cpp, ...
│   │   ├── config/                 # detector.yaml, stack.yaml
│   │   ├── launch/                 # rtbot_yolo_stack.launch.py
│   │   └── scripts/export_e2e.sh   # Engine export helper
│   └── interactive_tracker_cpp/    # FSM controller (C++) + keyboard node (Python)
│       ├── include/                # tracker_node.hpp
│       ├── src/                    # tracker_node.cpp, main.cpp
│       ├── config/                 # tracker_params.yaml
│       ├── launch/                 # tracker_system.launch.py
│       └── scripts/                # keyboard_command_node.py
├── sim_assets/
│   ├── worlds/                     # Custom Gazebo world files
│   ├── models/                     # Custom Gazebo models
│   └── scripts/patrol_entities.py  # Dynamic patrol script
├── benchmarks/
│   └── benchmark_tracker.py        # Performance measurement tool
├── docs/
│   └── benchmark_chart.py          # Chart generation script
├── scripts/env_setup.sh            # Environment setup helper
├── LICENSE                         # GPL-3.0-or-later
├── THIRD_PARTY_NOTICES.md          # Upstream attributions
└── README.md
```

---

## Third-Party Dependencies

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full details.

| Library | License |
|---------|--------|
| ROS2 Humble | Apache-2.0 |
| yolo_ros (mgonzs13) | GPL-3.0 |
| TurtleBot3 | Apache-2.0 |
| OpenCV | Apache-2.0 |
| TensorRT | NVIDIA SLA |
| ByteTrack | MIT |
| TensorRT-For-YOLO-Series | MIT |

---

## License

This project is licensed under the **GNU General Public License v3.0 or later**.
See [LICENSE](LICENSE) for the full text.

Note: This project links against TensorRT (NVIDIA proprietary). Users must accept the [NVIDIA TensorRT SLA](https://developer.nvidia.com/tensorrt) separately.

---

## Acknowledgements

- [mgonzs13/yolo_ros](https://github.com/mgonzs13/yolo_ros) — ByteTrack ROS2 integration
- [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3) — TurtleBot3 platform
- [ROBOTIS-GIT/turtlebot3_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations) — Gazebo simulation
- [triple-Mu/YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT) — TRT end2end export inspiration
- [YaoFANG1997/TensorRT-For-YOLO-Series](https://github.com/YaoFANG1997/TensorRT-For-YOLO-Series) — INT8 calibration reference
