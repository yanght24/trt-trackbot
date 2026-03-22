#!/bin/bash
# export_e2e.sh — 导出 end2end TensorRT engine (含 EfficientNMS_TRT 插件)
#
# 前置条件:
#   1. 已安装 TensorRT Python 绑定 (pip install tensorrt)
#   2. 已有 ONNX 模型文件
#   3. TensorRT-For-YOLO-Series-cuda-python 仓库可用
#
# 用法:
#   bash export_e2e.sh [onnx_path] [engine_path] [precision]
#
# 示例:
#   bash export_e2e.sh \
#     ~/models/yolo11n.onnx \
#     ~/models/yolo11n_e2e_fp16.engine \
#     fp16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Clone from: https://github.com/YaoFANG1997/TensorRT-For-YOLO-Series
EXPORT_TOOL="${TRT_YOLO_EXPORT_PY:-${HOME}/TensorRT-For-YOLO-Series-cuda-python/export.py}"

ONNX_PATH="${1:-${HOME}/models/yolo11n.onnx}"
ENGINE_PATH="${2:-${HOME}/models/yolo11n_e2e_fp16.engine}"
PRECISION="${3:-fp16}"

echo "═══════════════════════════════════════════════════"
echo "  End2end TensorRT Engine 导出"
echo "═══════════════════════════════════════════════════"
echo "  ONNX:      ${ONNX_PATH}"
echo "  Engine:    ${ENGINE_PATH}"
echo "  Precision: ${PRECISION}"
echo "  Export tool: ${EXPORT_TOOL}"
echo ""

if [ ! -f "${ONNX_PATH}" ]; then
  echo "ERROR: ONNX 文件不存在: ${ONNX_PATH}"
  echo ""
  echo "请先导出 ONNX:"
  echo "  pip install ultralytics"
  echo "  yolo export model=yolo11n.pt format=onnx opset=11 simplify=True"
  exit 1
fi

if [ ! -f "${EXPORT_TOOL}" ]; then
  echo "ERROR: 导出工具不存在: ${EXPORT_TOOL}"
  echo "请克隆 https://github.com/YaoFANG1997/TensorRT-For-YOLO-Series 并设置 TRT_YOLO_EXPORT_PY 环境变量"
  exit 1
fi

echo ">> 开始导出 end2end engine (--end2end --v8) ..."
echo ">> 这将在 engine 内部嵌入 EfficientNMS_TRT 插件"
echo ">> 输出张量: num, boxes, scores, classes"
echo ""

python3 "${EXPORT_TOOL}" \
  -o "${ONNX_PATH}" \
  -e "${ENGINE_PATH}" \
  -p "${PRECISION}" \
  --end2end \
  --v8 \
  --conf_thres 0.25 \
  --iou_thres 0.65 \
  --max_det 100

echo ""
echo "═══════════════════════════════════════════════════"
echo "  导出完成: ${ENGINE_PATH}"
echo ""
echo "  验证命令:"
echo "    python3 -c \""
echo "import tensorrt as trt"
echo "logger = trt.Logger(trt.Logger.INFO)"
echo "trt.init_libnvinfer_plugins(logger, '')"
echo "with open('${ENGINE_PATH}', 'rb') as f:"
echo "    runtime = trt.Runtime(logger)"
echo "    engine = runtime.deserialize_cuda_engine(f.read())"
echo "for i in range(engine.num_io_tensors):"
echo "    name = engine.get_tensor_name(i)"
echo "    shape = engine.get_tensor_shape(name)"
echo "    mode = engine.get_tensor_mode(name)"
echo "    print(f'  {name}: {shape}  {\"INPUT\" if mode == trt.TensorIOMode.INPUT else \"OUTPUT\"}')"
echo "\""
echo ""
echo "  终端 3 启动命令 (v2 全栈):"
echo "    source /path/to/trt-trackbot/scripts/env_setup.sh && ros2 launch rtbot_yolo_trt_cpp rtbot_yolo_stack.launch.py \\"
echo "      engine_path:=${ENGINE_PATH} \\"
echo "      input_image_topic:=/camera/image_raw \\"
echo "      threshold:=0.3 \\"
echo "      use_sim_time:=True"
echo "═══════���═══════════════════════════════════════════"
