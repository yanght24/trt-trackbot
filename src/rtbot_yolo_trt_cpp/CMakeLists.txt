cmake_minimum_required(VERSION 3.18)
project(rtbot_yolo_trt_cpp LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU compute capability at configure time via nvidia-smi.
# Override with: cmake -DCMAKE_CUDA_ARCHITECTURES=75 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _GPU_CC OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _GPU_CC_RET)
  if(_GPU_CC_RET EQUAL 0 AND _GPU_CC)
    string(REPLACE "." "" _GPU_CC_FLAT "${_GPU_CC}")
    # Take first GPU only
    string(REGEX MATCH "[0-9]+" _GPU_CC_FLAT "${_GPU_CC_FLAT}")
    set(CMAKE_CUDA_ARCHITECTURES ${_GPU_CC_FLAT})
  else()
    set(CMAKE_CUDA_ARCHITECTURES 86)  # safe fallback: Ampere
  endif()
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# -- Dependencies -----------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(yolo_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(message_filters REQUIRED)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include)
find_library(NVINFER_LIB nvinfer
  HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
  HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB)
  message(FATAL_ERROR "TensorRT not found. Install libnvinfer-dev.")
endif()
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib:     ${NVINFER_LIB}")

# -- Shared CUDA preprocess library (used by both v1 and v2 detectors) ------
add_library(cuda_preprocess STATIC
  src/cuda_preprocess.cu
)
target_include_directories(cuda_preprocess PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(cuda_preprocess CUDA::cudart)

# ====================================================================
# V2 targets (end2end full replacement)
# ====================================================================

# -- detector_node (v2: end2end TRT backend) ----------------------------
add_executable(detector_node
  src/main_detector.cpp
  src/detector_node.cpp
  src/trt_backend.cpp
)
target_include_directories(detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(detector_node
  rclcpp sensor_msgs std_msgs cv_bridge yolo_msgs
)
target_link_libraries(detector_node
  cuda_preprocess
  ${NVINFER_LIB}
  ${NVINFER_PLUGIN_LIB}
  CUDA::cudart
  ${OpenCV_LIBS}
)

# -- tracker_node (v2: C++ IoU tracker) ---------------------------------
add_executable(tracker_node
  src/main_tracker.cpp
  src/tracker_node.cpp
)
target_include_directories(tracker_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(tracker_node
  rclcpp yolo_msgs
)

# -- debug_node (v2: C++ OpenCV overlay) --------------------------------
add_executable(debug_node
  src/main_debug.cpp
  src/debug_node.cpp
)
target_include_directories(debug_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(debug_node
  rclcpp sensor_msgs cv_bridge yolo_msgs message_filters
)
target_link_libraries(debug_node
  ${OpenCV_LIBS}
)

# ====================================================================
# V1 detector (raw-head [1,84,N] + CPU NMS) -- for regression/benchmark
# Uses a standard YOLO .engine (NOT end2end). Independent target.
# ====================================================================
add_executable(detector_node_v1
  src/main_v1.cpp
  src/detector_node_v1.cpp
  src/trt_engine.cpp
  src/preprocess.cpp
  src/postprocess.cpp
)
target_include_directories(detector_node_v1 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(detector_node_v1
  rclcpp sensor_msgs std_msgs cv_bridge yolo_msgs
)
target_link_libraries(detector_node_v1
  cuda_preprocess
  ${NVINFER_LIB}
  CUDA::cudart
  ${OpenCV_LIBS}
)

# -- Install ------------------------------------------------------------
install(TARGETS detector_node tracker_node debug_node detector_node_v1
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# Install scripts
install(PROGRAMS scripts/export_e2e.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
