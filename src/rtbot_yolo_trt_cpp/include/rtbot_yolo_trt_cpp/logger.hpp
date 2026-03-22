// Copyright (c) 2026 yanght24 (GitHub: yanght24@gmail.com)
// SPDX-License-Identifier: GPL-3.0-or-later
// Project: https://github.com/yanght24/trt-trackbot
#pragma once
/// logger.hpp — 最小化的 TensorRT ILogger 实现，输出到 stderr
///
/// TensorRT 在创建 runtime/engine 时要求提供 ILogger。
/// 仅输出 WARNING 及以上级别的消息。

#include <NvInfer.h>
#include <iostream>
#include <string>

namespace rtbot_yolo_trt
{

class TrtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* msg) noexcept override
  {
    // 默认只打印 WARNING 及以上
    if (severity <= Severity::kWARNING) {
      const char* tag = "";
      switch (severity) {
        case Severity::kINTERNAL_ERROR: tag = "[TRT INTERNAL ERROR] "; break;
        case Severity::kERROR:          tag = "[TRT ERROR] ";          break;
        case Severity::kWARNING:        tag = "[TRT WARNING] ";        break;
        default: break;
      }
      std::cerr << tag << msg << std::endl;
    }
  }
};

}  // namespace rtbot_yolo_trt
