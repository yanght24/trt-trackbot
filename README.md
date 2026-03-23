<p align="center">
  <h1 align="center">🤖 trt-trackbot</h1>
  <p align="center">
    <b>ROS 2 · TensorRT · YOLO · ByteTrack · LiDAR</b><br/>
    End-to-end GPU tracking pipeline from camera to motor command
  </p>
  <p align="center">
    <a href="README_zh.md">🇨🇳 中文文档</a> ·
    <a href="#-quick-start">Quick Start</a> ·
    <a href="#-performance">Performance</a> ·
    <a href="#-architecture">Architecture</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/ROS2-Humble-blue?logo=ros" alt="ROS2"/>
    <img src="https://img.shields.io/badge/TensorRT-8.6%2B-green?logo=nvidia" alt="TensorRT"/>
    <img src="https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia" alt="CUDA"/>
    <img src="https://img.shields.io/badge/C%2B%2B-20-orange?logo=cplusplus" alt="C++20"/>
    <img src="https://img.shields.io/badge/License-GPL--3.0-red" alt="License"/>
    <img src="https://img.shields.io/github/stars/yanght24/trt-trackbot?style=social" alt="Stars"/>
  </p>
</p>

---

> **trt-trackbot** = TensorRT end2end detector (CPU-NMS-free) + C++ ByteTrack + FSM controller + LiDAR distance control
>
> Full GPU pipeline: `/camera/image_raw` → letterbox → TRT infer → EfficientNMS → ByteTrack → FSM → `/cmd_vel`

**trt-trackbot** is a ROS 2 robotics project that demonstrates how to build a real-time, interactive multi-object tracking system on a GPU-accelerated embedded/laptop platform. It is designed for researchers and engineers who want a **reproducible, benchmarkable baseline** for visual tracking on TurtleBot3 — from raw camera input all the way to closed-loop motor control.

The project evolved through three detection backends (Python TRT → C++ TRT v1 → C++ TRT v2 end2end), each with recorded benchmarks, so you can study the latency and power trade-offs of each optimization step. The controller supports keyboard-driven target locking, LiDAR-based distance regulation, and EMA-filtered yaw control — all in a single C++ node.

## ✨ Features

| | Feature | Detail |
|---|---------|--------|
| 🎯 | **Detector** | YOLOv11n FP16 TensorRT end2end (EfficientNMS_TRT embedded, no CPU NMS) |
| 🔄 | **Tracker** | ByteTrack — pure C++ implementation |
| 🧠 | **Controller** | C++ FSM: Manual / Locked / Searching |
| 📏 | **Distance** | LiDAR `/scan` → 20th-percentile ROI distance → PD velocity control |
| 🔢 | **Slot mapping** | ByteTrack large IDs (e.g. `1364`) → keyboard-friendly slots 1–9 |
| 🖼️ | **Overlay** | Real-time annotated image on `/tracker/overlay_image` |
| ⚡ | **Speed** | Python TRT 29 ms → C++ TRT v1 8.6 ms → **C++ TRT v2 2.4 ms** |

---

## 🔧 Prerequisites

| Component | Version | Notes |
|-----------|:-------:|-------|
| Ubuntu | **22.04** | |
| ROS 2 | **Humble** | Full desktop install |
| Gazebo | **Classic (11)** | Bundled with `turtlebot3-simulations` |
| CUDA | **12.x** | Required for TensorRT |
| TensorRT | **8.6+** | NVIDIA proprietary, install separately |
| OpenCV | **4.x** | via `ros-humble-cv-bridge` |
| Python | **3.10** | For benchmark / keyboard scripts |

### Step 1 — Install ROS 2 Humble

Follow the [official guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html), then:

```bash
sudo apt install \
  ros-humble-desktop \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-image-view \
  ros-humble-rqt-image-view
```

### Step 2 — Install TurtleBot3

```bash
sudo apt install ros-humble-turtlebot3 ros-humble-turtlebot3-simulations
echo 'export TURTLEBOT3_MODEL=waffle' >> ~/.bashrc
source ~/.bashrc
```

### Step 3 — Install TensorRT

Download the `.deb` installer matching your CUDA version from the [NVIDIA TensorRT download page](https://developer.nvidia.com/tensorrt).

Verify:
```bash
python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)"
```

### Step 4 — Build `yolo_msgs` (message definitions only)

This project uses `yolo_msgs/DetectionArray` for detector ↔ tracker ↔ controller communication. **Only the message package is needed** — the Python yolo_ros nodes are not used.

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone --depth 1 https://github.com/mgonzs13/yolo_ros.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select yolo_msgs
source install/setup.bash
```

### Step 5 — Export YOLO TensorRT Engine

The `.engine` file is hardware-specific (tied to your GPU SM version + TensorRT version) and **not included in this repo**. Build it on your target machine:

```bash
pip install ultralytics

python3 - <<'EOF'
from ultralytics import YOLO
m = YOLO('yolo11n.pt')
m.export(
    format='engine',
    half=True,       # FP16
    imgsz=640,
    device=0,
    simplify=True,
    nms=True,        # embeds EfficientNMS_TRT → end2end v2
)
# Output: yolo11n.engine  (in current directory)
EOF
```

Or use the provided script:
```bash
bash src/rtbot_yolo_trt_cpp/scripts/export_e2e.sh yolo11n.pt ~/engines/yolo11n_fp16.engine
```

> ⚠️ **Always rebuild the engine on the deployment machine.** An engine compiled on a different GPU will fail to load.

---

## 🔨 Build

```bash
# 1. Clone into a ROS 2 workspace
mkdir -p ~/trt_ws/src
cd ~/trt_ws/src
git clone https://github.com/yanght24/trt-trackbot.git .

# 2. Source ROS 2 and yolo_msgs
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 3. Build all packages
cd ~/trt_ws
colcon build --symlink-install

# 4. Source the workspace
source install/setup.bash
```

> 💡 **TensorRT linker errors?** Make sure `libnvinfer.so` is on your library path:
> ```bash
> export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
> ```

---

## 🚀 Quick Start

Each terminal below needs the same environment sourced first:

```bash
# ── Paste this in every new terminal ──────────────────────────
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle
source ~/trt_ws/install/setup.bash
# ──────────────────────────────────────────────────────────────
```

---

### Terminal 1 — Gazebo Simulation

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py use_sim_time:=True
```

> ⏳ Wait until the Gazebo window fully opens and the robot appears before starting Terminal 2+.

---

### Terminal 2 — Dynamic Patrol Entities *(optional)*

Spawns moving pedestrian and vehicle models for multi-target testing:

```bash
python3 ~/trt_ws/src/sim_assets/scripts/patrol_entities.py
```

> The custom Gazebo models (`casual_female`, `person_walking`, `hatchback_blue`, …) are included in `sim_assets/models/` — no extra download needed.

---

### Terminal 3 — TRT Detector + Tracker

```bash
ros2 launch rtbot_yolo_trt_cpp rtbot_yolo_stack.launch.py \
  engine_path:=/path/to/yolo11n_fp16.engine \
  input_image_topic:=/camera/image_raw \
  threshold:=0.3 \
  use_sim_time:=True
```

This starts three C++ nodes:

| Node | Role |
|------|------|
| `detector_node` | TRT inference (FP16 end2end) |
| `tracker_node` | ByteTrack multi-object tracking |
| `debug_node` | Raw detection overlay (optional) |

Expected output:
```
[detector_node]: TRT engine loaded: yolo11n_fp16.engine
[detector_node]: Input: 1x3x640x640, Output: [num_dets, boxes, scores, labels]
[tracker_node]: ByteTrack initialized — max_age=30, min_hits=1
```

---

### Terminal 4 — FSM Controller

```bash
ros2 launch interactive_tracker_cpp tracker_system.launch.py
```

Starts `TrackerManagerNode` with parameters from `config/tracker_params.yaml`.

Expected output:
```
[tracker_manager_node]: State: MANUAL
[tracker_manager_node]: Subscribed to /yolo/tracking, /scan, /camera/image_raw
```

---

### Terminal 5 — Keyboard Control

```bash
ros2 run interactive_tracker_cpp keyboard_command_node.py
```

The terminal shows a live slot table that refreshes every frame:

```
╔══════════════════════════════╗
║     Detected Targets         ║
╠══════════════════════════════╣
║  [1]  ID 1042   person       ║
║  [2]  ID 1364   person  ◀ LOCKED
║  [3]  ID 1071   car          ║
╚══════════════════════════════╝
State: LOCKED  |  Target ID: 1364  |  Dist: 1.34 m
```

**Keyboard bindings:**

| Key | Action |
|:---:|--------|
| `1` – `9` | Lock the target in slot N |
| `u` | Unlock → return to Manual |
| `s` | Searching mode (rotate in place) |
| `w` | Manual forward |
| `x` | Manual backward |
| `a` | Manual turn left |
| `d` | Manual turn right |
| `q` | Quit |

---

### Terminal 6 — Overlay Image Viewer *(optional)*

View the real-time annotated image with bounding boxes, slot numbers, distance, and lock indicator:

```bash
ros2 run rqt_image_view rqt_image_view /tracker/overlay_image
```

Or headless:
```bash
ros2 run image_view image_view --ros-args -r image:=/tracker/overlay_image
```

---

### Terminal 7 — Performance Benchmark *(optional)*

First lock a target in Terminal 5, then run:

```bash
# Must be run with ROS 2 sourced (same environment as other terminals)
python3 ~/trt_ws/src/benchmarks/benchmark_tracker.py \
  --tag v2_1920 \
  --duration 30 \
  --warmup 5
```

Results are saved to `benchmarks/v2_1920/benchmark.json`.

Compare multiple runs:
```bash
python3 ~/trt_ws/src/benchmarks/benchmark_tracker.py \
  --compare py_1920 v1_1920 v2_1920
```

Generate comparison charts:
```bash
python3 docs/benchmark_chart.py
# → docs/benchmark_fps.png
# → docs/benchmark_latency.png
```

---

## 📡 Topics Reference

| Topic | Type | Publisher | Description |
|-------|------|-----------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Gazebo | Raw camera input (1920×1080) |
| `/scan` | `sensor_msgs/LaserScan` | Gazebo | 360° LiDAR scan |
| `/yolo/detections` | `yolo_msgs/DetectionArray` | `detector_node` | Raw YOLO bounding boxes |
| `/yolo/tracking` | `yolo_msgs/DetectionArray` | `tracker_node` | ByteTrack output with stable IDs |
| `/user_command` | `std_msgs/String` | keyboard node | `lock:<ID>` / `slot:<N>` / `unlock` / `search` |
| `/manual_cmd_vel` | `geometry_msgs/Twist` | keyboard node | Manual velocity input |
| `/cmd_vel` | `geometry_msgs/Twist` | `TrackerManagerNode` | Output motor command |
| `/tracker/target_list` | `std_msgs/String` | `TrackerManagerNode` | JSON: slot→ID map + locked flag |
| `/tracker/overlay_image` | `sensor_msgs/Image` | `TrackerManagerNode` | Annotated camera image |
| `/tracker/target_distance` | `std_msgs/Float64` | `TrackerManagerNode` | LiDAR distance to locked target (m) |

---

## ⚙️ Configuration

### Controller — `src/interactive_tracker_cpp/config/tracker_params.yaml`

```yaml
interactive_tracker:
  ros__parameters:

    # ── LiDAR distance control (primary) ──────────────────────
    desired_distance_m:     1.2      # target following distance (m)
    distance_kp:            0.6      # proportional gain
    distance_deadband_m:    0.10     # ±10 cm dead zone

    # ── Yaw control ───────────────────────────────────────────
    yaw_kp:                 0.001
    yaw_deadband_px:        50.0     # pixels
    max_angular_z:          0.6      # rad/s

    # ── EMA smoothing ─────────────────────────────────────────
    center_x_alpha:         0.5      # target-center X filter
    angular_z_alpha:        0.7      # angular velocity filter

    # ── Target lifetime ───────────────────────────────────────
    target_lost_timeout_sec:  1.0
    slot_release_timeout_sec: 2.0

    # ── Control loop ──────────────────────────────────────────
    control_rate_hz:        20.0
```

### Detector — `src/rtbot_yolo_trt_cpp/config/stack.yaml`

```yaml
detector_node:
  ros__parameters:
    engine_path:      /path/to/yolo11n_fp16.engine   # ← set this
    conf_threshold:   0.3
```

---

## ❓ What's NOT in this repo

| Missing | Reason | How to get it |
|---------|--------|---------------|
| `*.engine` / `*.pt` / `*.onnx` | Binary, hardware-specific, >100 MB | Run `export_e2e.sh` or ultralytics export |
| TurtleBot3 workspace | Installed via `apt` | `sudo apt install ros-humble-turtlebot3-simulations` |
| Full `yolo_ros` Python nodes | Not needed — C++ nodes replace them | Build only `yolo_msgs` (Step 4 above) |
| `benchmarks/*/benchmark.json` | Raw data, regenerate locally | Run `benchmark_tracker.py` |
| `docs/benchmark_*.png` | Generated from JSON | Run `docs/benchmark_chart.py` |

---

## 📁 Project Structure

```
trt-trackbot/
├── src/
│   ├── rtbot_yolo_trt_cpp/                 # 🔍 TRT detector + C++ ByteTrack
│   │   ├── include/rtbot_yolo_trt_cpp/
│   │   │   ├── trt_backend.hpp             # TensorRT engine wrapper
│   │   │   ├── detector_node.hpp           # V2 end2end detector
│   │   │   ├── tracker_node.hpp            # C++ ByteTrack
│   │   │   └── ...
│   │   ├── src/
│   │   │   ├── trt_backend.cpp
│   │   │   ├── detector_node.cpp
│   │   │   ├── tracker_node.cpp
│   │   │   └── ...
│   │   ├── config/
│   │   │   ├── detector.yaml               # V1 raw-head params
│   │   │   └── stack.yaml                  # V2 end2end params ← edit engine_path here
│   │   ├── launch/
│   │   │   ├── rtbot_yolo_trt.launch.py    # V1: C++ detector + Python tracker
│   │   │   └── rtbot_yolo_stack.launch.py  # V2: fully C++ ← recommended
│   │   └── scripts/
│   │       └── export_e2e.sh               # Engine export helper
│   │
│   └── interactive_tracker_cpp/            # 🎮 FSM controller + keyboard
│       ├── include/interactive_tracker_cpp/
│       │   └── tracker_node.hpp
│       ├── src/
│       │   ├── tracker_node.cpp
│       │   └── main.cpp
│       ├── config/
│       │   └── tracker_params.yaml         # ← tune control params here
│       ├── launch/
│       │   └── tracker_system.launch.py
│       └── scripts/
│           └── keyboard_command_node.py
│
├── sim_assets/                             # 🌍 Gazebo simulation assets
│   ├── models/                             # Custom models (included)
│   │   ├── casual_female/
│   │   ├── person_walking/
│   │   ├── hatchback_blue/
│   │   └── ...
│   ├── worlds/
│   │   └── flat_tracking.world             # Open arena for testing
│   └── scripts/
│       └── patrol_entities.py
│
├── benchmarks/                             # 📈 Performance tools
│   └── benchmark_tracker.py
│
├── docs/                                   # 📊 Charts and assets
│   ├── benchmark_chart.py
│   ├── benchmark_fps.png
│   └── benchmark_latency.png
│
├── scripts/
│   └── env_setup.sh
│
├── LICENSE                                 # GPL-3.0-or-later
├── THIRD_PARTY_NOTICES.md
├── README.md                               # This file (English)
└── README_zh.md                            # Chinese version
```

---

## 📦 Third-Party Dependencies

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full attribution.

| Library | License | Usage |
|---------|:-------:|-------|
| [yolo_ros](https://github.com/mgonzs13/yolo_ros) | GPL-3.0 | `yolo_msgs` message definitions |
| [TurtleBot3](https://github.com/ROBOTIS-GIT/turtlebot3) | Apache-2.0 | Robot simulation |
| [TensorRT](https://developer.nvidia.com/tensorrt) | NVIDIA SLA | GPU inference runtime |
| ROS 2 Humble / rclcpp | Apache-2.0 | Robotics middleware |

---

## 🎬 Demo
The demo illustrates the full target-following workflow with three states: **Manual**, **Locked**, and **Searching**.

At the beginning, the robot operates in **Manual** mode, where the view is rotated manually to select the target. After selection, the system enters **Locked** mode and continuously tracks the person. When the target is temporarily occluded by the tree, the system switches to **Searching** mode and rotates to re-acquire the lost target. Once the person is detected again, it returns to **Locked** mode and resumes stable tracking.

<p align="center">
  <a href="docs/demo.mp4">
    <img src="docs/demo.gif" width="640" alt="Demo GIF">
  </a>
</p>

<p align="center">
  Click the preview image to watch or download the full demo video.
</p>

---

## 📊 Performance

> Hardware: **NVIDIA RTX 4070 Laptop GPU** · Resolution: **1920 × 1080** · 30 s steady-state recording

| | Backend | Mean FPS | Mean Latency | GPU Power | GPU Util | vs Baseline |
|:-:|:--------|:--------:|:------------:|:---------:|:--------:|:-----------:|
| ❌ | Python TRT *(baseline)* | 27.8 | 29.0 ms | 43.5 W | 41% | — |
| 🟡 | C++ TRT v1 *(raw-head)* | 29.5 | 8.6 ms | 26.1 W | 30% | latency **−70%** |
| ✅ | **C++ TRT v2 *(end2end)*** | **29.7** | **2.4 ms** | **25.8 W** | **26%** | latency **−92%**, power **−41%** |

> 🚀 v2 end2end moves all NMS inside TensorRT via `EfficientNMS_TRT` — zero CPU postprocessing.
> **12× faster than Python · 3.6× faster than C++ v1**

![FPS Comparison](docs/benchmark_fps.png)
![Latency Comparison](docs/benchmark_latency.png)

<details>
<summary>Reproduce benchmark charts</summary>

```bash
python3 docs/benchmark_chart.py
# → docs/benchmark_fps.png
# → docs/benchmark_latency.png
```
</details>

---

## Star History

<a href="https://www.star-history.com/?repos=yanght24%2Ftrt-trackbot&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=yanght24/trt-trackbot&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=yanght24/trt-trackbot&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=yanght24/trt-trackbot&type=date&legend=top-left" />
 </picture>
</a>

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 or later** — see [LICENSE](LICENSE).

> ⚠️ This project links against NVIDIA TensorRT (proprietary). Users must separately accept the [NVIDIA TensorRT Software License Agreement](https://developer.nvidia.com/nvidia-tensorrt-license-agreement).

---

## 🙏 Acknowledgements

- [mgonzs13/yolo_ros](https://github.com/mgonzs13/yolo_ros)
- [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3)
- [YaoFANG1997/TensorRT-For-YOLO-Series](https://github.com/YaoFANG1997/TensorRT-For-YOLO-Series)
