# Third-Party Notices

This project (trt-trackbot) is built upon the following open-source projects.
We gratefully acknowledge their authors and contributors.

---

## 1. TensorRT-For-YOLO-Series-cuda-python

- **Author:** triple-Mu
- **Repository:** https://github.com/triple-Mu/YOLOv8-TensorRT (and related repos)
- **License:** GPL-3.0
- **Usage:** End-to-end TensorRT engine export pipeline (`export.py`, `trt.py`, preprocessing/postprocessing utilities). The `rtbot_yolo_trt_cpp` package's TRT backend design is informed by this work.

---

## 2. yolo_ros

- **Author:** mgonzs13 and contributors
- **Repository:** https://github.com/mgonzs13/yolo_ros
- **License:** GPL-3.0
- **Usage:** ROS2 message definitions (`yolo_msgs`) and ByteTrack integration. The `/yolo/tracking` topic and `Detection`/`DetectionArray` message types originate from this package.

---

## 3. ByteTrack

- **Author:** ifzhang (Zhang Yifu) and contributors
- **Repository:** https://github.com/ifzhang/ByteTrack
- **License:** MIT
- **Usage:** Multi-object tracking algorithm used as the core tracking backend via `yolo_ros`.

---

## 4. TurtleBot3

- **Author:** ROBOTIS Co., Ltd.
- **Repository:** https://github.com/ROBOTIS-GIT/turtlebot3
- **License:** Apache-2.0
- **Usage:** Robot simulation model (Waffle) and Gazebo launch files used in `ws_turt/` for testing and demo.

---

## 5. ROS2 Humble / rclcpp

- **Author:** Open Robotics and contributors
- **Repository:** https://github.com/ros2
- **License:** Apache-2.0
- **Usage:** Robotics middleware. All ROS2 nodes depend on `rclcpp`, `sensor_msgs`, `geometry_msgs`, and related packages.

---

## 6. NVIDIA TensorRT

- **Author:** NVIDIA Corporation
- **License:** [NVIDIA TensorRT Software License Agreement](https://developer.nvidia.com/nvidia-tensorrt-license-agreement)
- **Usage:** GPU inference runtime. The `rtbot_yolo_trt_cpp` package links against TensorRT shared libraries (`libnvinfer`, `libnvonnxparser`).
- **Note:** TensorRT is a proprietary SDK. Users must accept NVIDIA's license agreement separately.

---

## 7. OpenCV

- **Author:** OpenCV Foundation and contributors
- **Repository:** https://github.com/opencv/opencv
- **License:** Apache-2.0
- **Usage:** Image preprocessing and visualization (resize, letterbox, overlay drawing).

---

## 8. cv_bridge / image_transport

- **Author:** Open Robotics and contributors
- **Repository:** https://github.com/ros-perception/vision_opencv
- **License:** BSD-3-Clause
- **Usage:** Conversion between `sensor_msgs/Image` and `cv::Mat`.

---

*If you believe your project has been inadvertently omitted or misattributed, please open an issue.*
