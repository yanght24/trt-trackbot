#!/usr/bin/env python3
"""
benchmark_tracker.py — 自动化性能基准测试 v3

v3 新增:
  - 订阅 /yolo/detections（检测器原始输出），独立于 /yolo/tracking
  - 检测延迟: 通过 header.stamp 匹配 image_raw → detections 的端到端延迟
  - 检测 FPS: 检测器吞吐量（不受 tracking_node 同步瓶颈影响）
  - 对比表新增检测器性能和资源效率指标

v2 修正:
  1. 延迟统计改为 wall-clock 帧间隔推算，不依赖 sim_time/header.stamp
  2. GPU/CPU 改为每秒采样，报告均值和 p95
  3. 控制质量指标只统计 Locked 状态
  4. 开始录制前自动等待 warmup（等到 Locked 状态 + N帧稳定）
  5. 角速度抖动只统计 |angular_z| > 阈值的符号翻转

用法:
  # 先启动完整系统，锁定一个目标，然后：
  python3 benchmark_tracker.py --tag pytorch_baseline --duration 30

  # TensorRT 优化后：
  python3 benchmark_tracker.py --tag tensorrt_fp16 --duration 30

  # 对比：
  python3 benchmark_tracker.py --compare pytorch_baseline tensorrt_fp16
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64
from yolo_msgs.msg import DetectionArray


class BenchmarkNode(Node):
    def __init__(self, duration: float, warmup: float):
        super().__init__('benchmark_node')
        self.duration = duration
        self.warmup = warmup

        # Phase tracking
        self._phase = 'warmup'  # warmup -> recording -> done
        self._warmup_start = time.monotonic()
        self._recording_start = None

        # Current state from /tracker/state
        self._current_state = ''

        # ── YOLO 帧率 (wall-clock) ────────────────────────────────────
        self._yolo_wall_times = []           # monotonic timestamps of each frame arrival
        self._yolo_det_counts = []

        # ── 相机帧率 + stamp→wall 映射 ─────────────────────────────────
        self._image_wall_times = []
        # 用于检测延迟: header.stamp (nanoseconds) → wall-clock arrival
        self._image_stamp_to_wall = {}

        # ── 检测器原始输出 (不经过 tracking_node 同步) ────────────────
        self._det_wall_times = []       # wall-clock 到达时刻
        self._det_latencies = []        # 每帧: detections到达 - 对应image到达 (秒)
        self._det_counts = []           # 每帧检测数量

        # ── 控制输出 (只在 Locked 状态收集) ───────────────────────────
        self._locked_cmd_linear = []
        self._locked_cmd_angular = []

        # ── 距离 (只在 Locked 状态收集) ───────────────────────────────
        self._locked_distances = []

        # ── 状态时间分布 ──────────────────────────────────────────────
        self._state_samples = []

        # ── GPU/CPU 采样 (每秒一次) ───────────────────────────────────
        self._gpu_samples = []
        self._cpu_samples = []

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(DetectionArray, '/yolo/tracking', self._yolo_cb, sensor_qos)
        self.create_subscription(DetectionArray, '/yolo/detections', self._det_cb, sensor_qos)
        self.create_subscription(Image, '/camera/image_raw', self._image_cb, sensor_qos)
        self.create_subscription(Twist, '/cmd_vel', self._cmdvel_cb, 10)
        self.create_subscription(Float64, '/tracker/target_distance', self._dist_cb, 10)
        self.create_subscription(String, '/tracker/state', self._state_cb, 10)

        self._tick_timer = self.create_timer(1.0, self._tick)

        self.get_logger().info(
            f'Benchmark: warmup={warmup:.0f}s, then record={duration:.0f}s. '
            f'Waiting for Locked state...')

    # ── 内部状态 ───────────────────────────────────────────────────────

    def _is_recording(self):
        return self._phase == 'recording'

    # ── 订阅回调 ───────────────────────────────────────────────────────

    def _yolo_cb(self, msg):
        if not self._is_recording():
            return
        self._yolo_wall_times.append(time.monotonic())
        self._yolo_det_counts.append(len(msg.detections))

    def _image_cb(self, msg):
        now = time.monotonic()
        # Always record stamp→wall mapping (needed for detection latency)
        stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self._image_stamp_to_wall[stamp_ns] = now
        # Limit dict size to avoid unbounded growth
        if len(self._image_stamp_to_wall) > 500:
            oldest_keys = sorted(self._image_stamp_to_wall.keys())[:200]
            for k in oldest_keys:
                del self._image_stamp_to_wall[k]
        if not self._is_recording():
            return
        self._image_wall_times.append(now)

    def _det_cb(self, msg):
        """Callback for /yolo/detections (raw detector output, before tracking)."""
        if not self._is_recording():
            return
        now = time.monotonic()
        self._det_wall_times.append(now)
        self._det_counts.append(len(msg.detections))
        # Match header.stamp to image arrival for latency calculation
        stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        img_wall = self._image_stamp_to_wall.get(stamp_ns)
        if img_wall is not None:
            latency = now - img_wall
            if latency >= 0:
                self._det_latencies.append(latency)

    def _cmdvel_cb(self, msg):
        if not self._is_recording():
            return
        if self._current_state == 'Locked':
            self._locked_cmd_linear.append(msg.linear.x)
            self._locked_cmd_angular.append(msg.angular.z)

    def _dist_cb(self, msg):
        if not self._is_recording():
            return
        if self._current_state == 'Locked':
            self._locked_distances.append(msg.data)

    def _state_cb(self, msg):
        self._current_state = msg.data
        if self._is_recording():
            self._state_samples.append(msg.data)

    # ── 每秒 tick ──────────────────────────────────────────────────────

    def _tick(self):
        now = time.monotonic()

        if self._phase == 'warmup':
            elapsed = now - self._warmup_start
            if self._current_state == 'Locked' and elapsed >= self.warmup:
                self._phase = 'recording'
                self._recording_start = now
                self.get_logger().info(
                    f'Warmup done ({elapsed:.1f}s). Recording started.')
            else:
                state_str = self._current_state or '(waiting)'
                self.get_logger().info(
                    f'Warmup {elapsed:.0f}/{self.warmup:.0f}s  state={state_str}')
            return

        if self._phase == 'recording':
            elapsed = now - self._recording_start
            remaining = self.duration - elapsed

            # 每秒采样 GPU/CPU
            self._gpu_samples.append(_collect_gpu_once())
            self._cpu_samples.append(_collect_cpu_once())

            if remaining > 0:
                self.get_logger().info(
                    f'Recording {elapsed:.0f}/{self.duration:.0f}s  '
                    f'track={len(self._yolo_wall_times)}  '
                    f'det={len(self._det_wall_times)}  '
                    f'lat={len(self._det_latencies)}  '
                    f'dist={len(self._locked_distances)}')
            else:
                self._phase = 'done'
                self.get_logger().info('Recording complete.')
                raise SystemExit(0)

    # ── 结果计算 ───────────────────────────────────────────────────────

    def compute_results(self):
        r = {}

        # 1. YOLO FPS (wall-clock frame intervals)
        if len(self._yolo_wall_times) > 1:
            intervals = [self._yolo_wall_times[i+1] - self._yolo_wall_times[i]
                         for i in range(len(self._yolo_wall_times)-1)]
            intervals.sort()
            mean_interval = sum(intervals) / len(intervals)
            r['yolo_fps_mean'] = round(1.0 / mean_interval, 1)
            r['yolo_fps_p50'] = round(1.0 / intervals[len(intervals)//2], 1)
            r['yolo_total_frames'] = len(self._yolo_wall_times)
            # Frame interval as proxy for per-frame processing time
            r['yolo_frame_interval_mean_ms'] = round(mean_interval * 1000, 1)
            r['yolo_frame_interval_p95_ms'] = round(
                intervals[int(len(intervals)*0.95)] * 1000, 1)
        else:
            r['yolo_fps_mean'] = 0
            r['yolo_total_frames'] = len(self._yolo_wall_times)

        # 2. Camera FPS
        if len(self._image_wall_times) > 1:
            intervals = [self._image_wall_times[i+1] - self._image_wall_times[i]
                         for i in range(len(self._image_wall_times)-1)]
            r['camera_fps'] = round(1.0 / (sum(intervals)/len(intervals)), 1)

        # 3. Detections per frame
        if self._yolo_det_counts:
            r['avg_detections_per_frame'] = round(
                sum(self._yolo_det_counts) / len(self._yolo_det_counts), 2)

        # 3b. Detection node performance (raw /yolo/detections, before tracking)
        if len(self._det_wall_times) > 1:
            det_intervals = [self._det_wall_times[i+1] - self._det_wall_times[i]
                             for i in range(len(self._det_wall_times)-1)]
            det_intervals.sort()
            det_mean_interval = sum(det_intervals) / len(det_intervals)
            r['det_fps_mean'] = round(1.0 / det_mean_interval, 1)
            r['det_fps_p50'] = round(1.0 / det_intervals[len(det_intervals)//2], 1)
            r['det_total_frames'] = len(self._det_wall_times)
            r['det_interval_mean_ms'] = round(det_mean_interval * 1000, 1)
            r['det_interval_p95_ms'] = round(
                det_intervals[int(len(det_intervals)*0.95)] * 1000, 1)
        else:
            r['det_fps_mean'] = 0
            r['det_total_frames'] = len(self._det_wall_times)

        # 3c. Detection latency (image_raw arrival → detections arrival, matched by stamp)
        if self._det_latencies:
            lats = sorted(self._det_latencies)
            r['det_latency_mean_ms'] = round(sum(lats) / len(lats) * 1000, 1)
            r['det_latency_p50_ms'] = round(lats[len(lats)//2] * 1000, 1)
            r['det_latency_p95_ms'] = round(lats[int(len(lats)*0.95)] * 1000, 1)
            r['det_latency_min_ms'] = round(lats[0] * 1000, 1)
            r['det_latency_max_ms'] = round(lats[-1] * 1000, 1)
            r['det_latency_matched_frames'] = len(lats)

        # 4. Distance stability (Locked only)
        valid = [d for d in self._locked_distances if d > 0]
        r['distance_total_samples'] = len(self._locked_distances)
        r['distance_valid_samples'] = len(valid)
        if valid:
            mean_d = sum(valid) / len(valid)
            var = sum((d - mean_d)**2 for d in valid) / len(valid)
            r['distance_mean_m'] = round(mean_d, 3)
            r['distance_std_m'] = round(var**0.5, 3)
            r['distance_min_m'] = round(min(valid), 3)
            r['distance_max_m'] = round(max(valid), 3)
            r['distance_valid_ratio'] = round(len(valid) / len(self._locked_distances), 3)

        # 5. Angular jitter (Locked only, threshold=0.05)
        JITTER_THRESHOLD = 0.05
        if self._locked_cmd_angular:
            # Only count sign changes between samples that exceed threshold
            significant = [a for a in self._locked_cmd_angular if abs(a) > JITTER_THRESHOLD]
            sign_changes = sum(
                1 for i in range(1, len(significant))
                if significant[i] * significant[i-1] < 0)
            r['angular_z_mean_abs'] = round(
                sum(abs(a) for a in self._locked_cmd_angular) / len(self._locked_cmd_angular), 4)
            r['angular_z_max_abs'] = round(max(abs(a) for a in self._locked_cmd_angular), 4)
            r['angular_significant_samples'] = len(significant)
            r['angular_sign_changes'] = sign_changes
            r['angular_jitter_rate'] = round(
                sign_changes / max(1, len(significant)), 4)

        # 6. Linear velocity (Locked only)
        if self._locked_cmd_linear:
            r['linear_x_mean'] = round(
                sum(self._locked_cmd_linear) / len(self._locked_cmd_linear), 4)
            r['linear_x_max'] = round(max(self._locked_cmd_linear), 4)
            r['linear_x_min'] = round(min(self._locked_cmd_linear), 4)
            r['locked_cmd_samples'] = len(self._locked_cmd_linear)

        # 7. State distribution
        if self._state_samples:
            from collections import Counter
            counts = Counter(self._state_samples)
            total = len(self._state_samples)
            r['state_distribution'] = {k: round(v/total*100, 1) for k, v in counts.items()}

        # 8. GPU stats (averaged over all samples)
        r['gpu'] = _aggregate_samples(self._gpu_samples)

        # 9. CPU stats (averaged over all samples)
        r['cpu'] = _aggregate_samples(self._cpu_samples)

        r['recording_duration_sec'] = round(self.duration, 1)
        r['warmup_sec'] = round(self.warmup, 1)
        r['jitter_threshold'] = JITTER_THRESHOLD

        return r


# ── GPU/CPU 采样工具 ──────────────────────────────────────────────────

def _collect_gpu_once():
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], text=True, timeout=3).strip()
        parts = [p.strip() for p in out.split(',')]
        return {
            'gpu_util_pct': int(parts[0]),
            'gpu_mem_used_mb': int(parts[1]),
            'gpu_mem_total_mb': int(parts[2]),
            'gpu_power_w': float(parts[3]),
            'gpu_temp_c': int(parts[4]),
        }
    except Exception:
        return None


def _collect_cpu_once():
    try:
        load1, _, _ = os.getloadavg()
        cores = os.cpu_count() or 1
        return {'cpu_load_1min': round(load1, 2), 'cpu_cores': cores}
    except Exception:
        return None


def _aggregate_samples(samples):
    """Aggregate numeric samples: compute mean and p95 for each key."""
    valid = [s for s in samples if s is not None]
    if not valid:
        return {}

    result = {}
    # Get GPU name from nvidia-smi once
    try:
        name = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name', '--format=csv,noheader'
        ], text=True, timeout=3).strip()
        result['name'] = name
    except Exception:
        pass

    for key in valid[0]:
        vals = [s[key] for s in valid if key in s and isinstance(s[key], (int, float))]
        if not vals:
            continue
        vals.sort()
        result[f'{key}_mean'] = round(sum(vals) / len(vals), 1)
        result[f'{key}_p95'] = round(vals[int(len(vals) * 0.95)], 1)
        result[f'{key}_max'] = round(vals[-1], 1)
    return result


# ── 报告打印 ──────────────────────────────────────────────────────────

def _print_report(r):
    tag = r.get('tag', '?')
    print('\n' + '=' * 64)
    print(f'  BENCHMARK REPORT: {tag}')
    print('=' * 64)
    print(f'  Warmup:   {r.get("warmup_sec", 0)} s')
    print(f'  Duration: {r.get("recording_duration_sec", 0)} s')
    print()
    print('  ── YOLO 系统级性能 (tracking输出, wall-clock) ──')
    print(f'  Tracking FPS:        {r.get("yolo_fps_mean", 0)} (mean)'
          f'  {r.get("yolo_fps_p50", "")} (p50)')
    print(f'  Tracking frames:     {r.get("yolo_total_frames", 0)}')
    print(f'  Tracking interval:   {r.get("yolo_frame_interval_mean_ms", "?")} ms (mean)'
          f'  {r.get("yolo_frame_interval_p95_ms", "?")} ms (p95)')
    print(f'  Avg dets/frame:      {r.get("avg_detections_per_frame", 0)}')
    print(f'  Camera FPS:          {r.get("camera_fps", "?")}')
    print()
    print('  ── 检测器性能 (detector原始输出, 不含tracking) ──')
    print(f'  Detector FPS:        {r.get("det_fps_mean", "N/A")} (mean)'
          f'  {r.get("det_fps_p50", "")} (p50)')
    print(f'  Detector frames:     {r.get("det_total_frames", "N/A")}')
    print(f'  Detector interval:   {r.get("det_interval_mean_ms", "N/A")} ms (mean)'
          f'  {r.get("det_interval_p95_ms", "N/A")} ms (p95)')
    print(f'  Detection latency:   {r.get("det_latency_mean_ms", "N/A")} ms (mean)'
          f'  {r.get("det_latency_p50_ms", "N/A")} ms (p50)'
          f'  {r.get("det_latency_p95_ms", "N/A")} ms (p95)')
    print(f'  Latency range:       [{r.get("det_latency_min_ms", "N/A")}'
          f', {r.get("det_latency_max_ms", "N/A")}] ms'
          f'  (matched {r.get("det_latency_matched_frames", 0)} frames)')
    print()
    print('  ── 系统资源 (录制期间均值) ──')
    gpu = r.get('gpu', {})
    cpu = r.get('cpu', {})
    print(f'  GPU:                 {gpu.get("name", "N/A")}')
    print(f'  GPU Util:            {gpu.get("gpu_util_pct_mean", "?")}%'
          f'  (p95: {gpu.get("gpu_util_pct_p95", "?")}%)')
    print(f'  GPU Memory:          {gpu.get("gpu_mem_used_mb_mean", "?")}'
          f' / {gpu.get("gpu_mem_total_mb_mean", "?")} MB')
    print(f'  GPU Power:           {gpu.get("gpu_power_w_mean", "?")} W'
          f'  (p95: {gpu.get("gpu_power_w_p95", "?")} W)')
    print(f'  GPU Temp:            {gpu.get("gpu_temp_c_mean", "?")} °C')
    print(f'  CPU Load:            {cpu.get("cpu_load_1min_mean", "?")}')
    print()
    print('  ── 控制质量 (仅 Locked 状态) ──')
    print(f'  Locked cmd samples:  {r.get("locked_cmd_samples", 0)}')
    print(f'  Distance (mean±σ):   {r.get("distance_mean_m", "?")} ± {r.get("distance_std_m", "?")} m')
    print(f'  Distance range:      [{r.get("distance_min_m", "?")}, {r.get("distance_max_m", "?")}] m')
    print(f'  LiDAR valid rate:    '
          f'{r.get("distance_valid_ratio", 0)*100:.1f}%'
          f'  ({r.get("distance_valid_samples", 0)}/{r.get("distance_total_samples", 0)})')
    print(f'  Angular |z| mean:    {r.get("angular_z_mean_abs", "?")} rad/s')
    print(f'  Angular jitter:      {r.get("angular_sign_changes", "?")} sign changes'
          f'  / {r.get("angular_significant_samples", "?")} significant'
          f'  = {(r.get("angular_jitter_rate", 0))*100:.1f}%'
          f'  (threshold={r.get("jitter_threshold", "?")} rad/s)')
    print(f'  Linear x (mean):     {r.get("linear_x_mean", "?")} m/s')
    sd = r.get('state_distribution', {})
    if sd:
        print(f'  State distribution:  {sd}')
    print('=' * 64)


# ── 对比 ──────────────────────────────────────────────────────────────

def compare(tags):
    base = Path.home() / 'Project' / 'rtbot' / 'benchmarks'
    datasets = []
    for tag in tags:
        p = base / tag / 'benchmark.json'
        if not p.exists():
            print(f'ERROR: {p} not found'); sys.exit(1)
        with open(p) as f:
            datasets.append(json.load(f))

    keys = [
        # ── 检测器性能 (核心对比) ──
        ('det_latency_mean_ms',         'Det latency (mean)',    'ms', 'lower'),
        ('det_latency_p50_ms',          'Det latency (p50)',     'ms', 'lower'),
        ('det_latency_p95_ms',          'Det latency (p95)',     'ms', 'lower'),
        ('det_fps_mean',                'Detector FPS',          '',   'higher'),
        ('det_interval_mean_ms',        'Det interval (mean)',   'ms', 'lower'),
        # ── 系统级性能 ──
        ('yolo_fps_mean',               'Tracking FPS (系统级)', '',   'higher'),
        ('yolo_frame_interval_mean_ms', 'Tracking interval',     'ms', 'lower'),
        ('yolo_frame_interval_p95_ms',  'Tracking intv (p95)',   'ms', 'lower'),
        # ── 控制质量 ──
        ('distance_std_m',              'Distance σ (Locked)',   'm',  'lower'),
        ('distance_valid_ratio',        'LiDAR valid rate',      '',   'higher'),
        ('angular_jitter_rate',         'Angular jitter',        '',   'lower'),
    ]

    print('\n' + '=' * 80)
    print('  PERFORMANCE COMPARISON')
    print('=' * 80)
    hdr = f'  {"Metric":<28}'
    for d in datasets:
        hdr += f'  {d["tag"]:>18}'
    hdr += f'  {"Change":>12}'
    print(hdr)
    print('  ' + '-' * 76)

    for key, label, unit, direction in keys:
        row = f'  {label:<28}'
        vals = []
        for d in datasets:
            v = d.get(key, 'N/A')
            vals.append(v)
            if isinstance(v, (int, float)):
                row += f'  {v:>16.1f}{unit}'
            else:
                row += f'  {str(v):>18}'
        if (len(vals) == 2
                and isinstance(vals[0], (int, float))
                and isinstance(vals[1], (int, float))
                and vals[0] != 0):
            pct = (vals[1] - vals[0]) / abs(vals[0]) * 100
            ok = (pct > 0 and direction == 'higher') or (pct < 0 and direction == 'lower')
            row += f'  {pct:>+7.1f}% {"✓" if ok else "✗"}'
        print(row)

    # ── 资源效率对比 ──
    print()
    print('  ── 资源效率 ──')
    res_keys = [
        ('gpu.gpu_util_pct_mean',    'GPU Util',     '%',  'lower'),
        ('gpu.gpu_mem_used_mb_mean', 'GPU Memory',   'MB', 'lower'),
        ('gpu.gpu_power_w_mean',     'GPU Power',    'W',  'lower'),
        ('gpu.gpu_temp_c_mean',      'GPU Temp',     '°C', 'lower'),
        ('cpu.cpu_load_1min_mean',   'CPU Load',     '',   'lower'),
    ]
    for dotkey, label, unit, direction in res_keys:
        parts = dotkey.split('.')
        row = f'  {label:<28}'
        vals = []
        for d in datasets:
            v = d.get(parts[0], {}).get(parts[1], 'N/A') if len(parts) == 2 else d.get(dotkey, 'N/A')
            vals.append(v)
            if isinstance(v, (int, float)):
                row += f'  {v:>16.1f}{unit}'
            else:
                row += f'  {str(v):>18}'
        if (len(vals) == 2
                and isinstance(vals[0], (int, float))
                and isinstance(vals[1], (int, float))
                and vals[0] != 0):
            pct = (vals[1] - vals[0]) / abs(vals[0]) * 100
            ok = (pct > 0 and direction == 'higher') or (pct < 0 and direction == 'lower')
            row += f'  {pct:>+7.1f}% {"✓" if ok else "✗"}'
        print(row)

    print('=' * 80)


# ── 主入口 ────────────────────────────────────────────────────────────

def run_benchmark(tag, duration, warmup):
    out_dir = Path.home() / 'Project' / 'rtbot' / 'benchmarks' / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = BenchmarkNode(duration, warmup)
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass

    results = node.compute_results()
    results['tag'] = tag
    results['timestamp'] = datetime.now().isoformat()

    out_file = out_dir / 'benchmark.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    node.destroy_node()
    rclpy.shutdown()

    _print_report(results)
    print(f'\n  Saved: {out_file}\n')


def main():
    p = argparse.ArgumentParser(description='Tracker performance benchmark v3')
    p.add_argument('--tag', default='baseline', help='e.g. pytorch_baseline, tensorrt_fp16')
    p.add_argument('--duration', type=float, default=30.0, help='Recording duration (seconds)')
    p.add_argument('--warmup', type=float, default=5.0,
                   help='Wait this many seconds in Locked state before recording')
    p.add_argument('--compare', nargs='+', metavar='TAG', help='Compare benchmark results')
    args = p.parse_args()

    if args.compare:
        compare(args.compare)
    else:
        run_benchmark(args.tag, args.duration, args.warmup)


if __name__ == '__main__':
    main()
