cmake_minimum_required(VERSION 3.8)
project(interactive_tracker_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yolo_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(tracker_manager_node
  src/main.cpp
  src/tracker_node.cpp
)

target_include_directories(tracker_manager_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(tracker_manager_node
  rclcpp
  geometry_msgs
  std_msgs
  sensor_msgs
  yolo_msgs
  cv_bridge
)

target_link_libraries(tracker_manager_node
  ${OpenCV_LIBS}
)

install(TARGETS tracker_manager_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/keyboard_command_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME keyboard_command_node
)

ament_package()
