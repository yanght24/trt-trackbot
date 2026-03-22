#!/bin/bash
# env_setup.sh — trt-trackbot 环境配置脚本
#
# 用法: source /path/to/trt-trackbot/scripts/env_setup.sh
# 请根据你的实际安装路径修改 WORKSPACE_ROOT

# ── 用户配置区域 ──────────────────────────────────────────────────────────────
# 修改为你的 trt-trackbot 工作空间根目录（克隆后的路径）
WORKSPACE_ROOT="${HOME}/trt-trackbot"

# TurtleBot3 安装工作空间路径（ws_turt 或系统安装）
# 如果使用 apt 安装的 turtlebot3, 则无需此行
TURTLEBOT3_WS="${HOME}/ws_turt"   # 如果不需要，注释掉此行
# ── 用户配置区域结束 ─────────────────────────────────────────────────────────

# 加载 ROS 2 Humble 基础环境
source /opt/ros/humble/setup.bash

# Gazebo 资源路径
if [ -f "/usr/share/gazebo/setup.sh" ]; then
    source /usr/share/gazebo/setup.sh
fi

# 核心环境变量
export TURTLEBOT3_MODEL=waffle
export ROS_DOMAIN_ID=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 加载 TurtleBot3 工作空间 (如使用源码编译版本)
if [ -f "${TURTLEBOT3_WS}/install/setup.bash" ]; then
    source "${TURTLEBOT3_WS}/install/setup.bash"
fi

# 加载主工作空间（colcon build 后执行）
if [ -f "${WORKSPACE_ROOT}/install/setup.bash" ]; then
    source "${WORKSPACE_ROOT}/install/setup.bash"
fi

# Gazebo 仿真资产路径
export GAZEBO_MODEL_PATH="${WORKSPACE_ROOT}/sim_assets/models:${HOME}/.gazebo/models:${GAZEBO_MODEL_PATH}"
export GAZEBO_RESOURCE_PATH="${WORKSPACE_ROOT}/sim_assets/worlds:${GAZEBO_RESOURCE_PATH}"

export PS1="\[\033[01;32m\][trt-trackbot]\[\033[00m\] \[\033[01;34m\]\w\[\033[00m\]$ "

echo "[trt-trackbot] Environment loaded from: ${WORKSPACE_ROOT}"
