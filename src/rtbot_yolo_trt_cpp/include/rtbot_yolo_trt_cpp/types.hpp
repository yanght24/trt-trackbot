// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// types.hpp — v1 兼容层，实际类型定义已统一到 common.hpp
///
/// v1 的 trt_engine.hpp / postprocess.hpp / preprocess.hpp 原本 include 此文件。
/// 现在直接转发到 common.hpp，消除 Detection / PreprocessInfo 的重复定义。

#include "rtbot_yolo_trt_cpp/common.hpp"
