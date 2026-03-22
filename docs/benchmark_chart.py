#!/usr/bin/env python3
"""Generate performance comparison bar charts from benchmark JSON files.

Usage:
    python3 docs/benchmark_chart.py

Outputs:
    docs/benchmark_fps.png
    docs/benchmark_latency.png"""

import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BENCH_DIR = pathlib.Path(__file__).parent.parent / 'benchmarks'

# ── Load all benchmark files ──────────────────────────────────────────────────
records = {}
for tag in ['py_1920', 'v1_1920', 'v2_1920',
            'pytorch_baseline', 'tensorrt_cpp_detector',
            'tensorrt_cpp_detector_v1', 'tensorrt_cpp_stack_v2']:
    p = BENCH_DIR / tag / 'benchmark.json'
    if p.exists():
        with open(p) as f:
            records[tag] = json.load(f)

# ── Select the 1920-width series for the main chart ──────────────────────────
SERIES = [
    ('py_1920',  'Python TRT\n(1920px)'),
    ('v1_1920',  'C++ TRT v1\nraw-head (1920px)'),
    ('v2_1920',  'C++ TRT v2\nend2end (1920px)'),
]

labels   = [s[1] for s in SERIES if s[0] in records]
fps_vals = [records[s[0]]['det_fps_mean'] for s in SERIES if s[0] in records]
lat_vals = [records[s[0]]['det_latency_mean_ms'] for s in SERIES if s[0] in records]

colors = ['#4C72B0', '#55A868', '#C44E52']

# ── FPS chart ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, fps_vals, color=colors, width=0.5, edgecolor='white', linewidth=0.8)
ax.bar_label(bars, fmt='%.1f fps', padding=4, fontsize=11, fontweight='bold')
ax.set_ylabel('Frames per Second (fps)', fontsize=12)
ax.set_title('YOLO Detection Throughput — Python vs C++ TRT\n(RTX 4070 Laptop, 1920×1080)', fontsize=13)
ax.set_ylim(0, max(fps_vals) * 1.25)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(BENCH_DIR.parent / 'docs' / 'benchmark_fps.png', dpi=150)
print('Saved docs/benchmark_fps.png')
plt.close()

# ── Latency chart ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, lat_vals, color=colors, width=0.5, edgecolor='white', linewidth=0.8)
ax.bar_label(bars, fmt='%.1f ms', padding=4, fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Latency (ms)', fontsize=12)
ax.set_title('YOLO Detection Latency — Python vs C++ TRT\n(RTX 4070 Laptop, 1920×1080)', fontsize=13)
ax.set_ylim(0, max(lat_vals) * 1.3)
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(BENCH_DIR.parent / 'docs' / 'benchmark_latency.png', dpi=150)
print('Saved docs/benchmark_latency.png')
plt.close()

print('Done.')
