#!/usr/bin/env python3
"""
tests/benchmark_intrusion_comparison.py
=========================================
Standalone benchmark comparing IntrusionDetectionPipeline (baseline)
against NewIntrusionPipeline (OpenVINO-accelerated).

Usage (identical to how you use the existing intrusion pipeline):
  # With an uploaded video file
  python tests/benchmark_intrusion_comparison.py --source storage/uploads/<file.mp4> --frames 300

  # With an RTSP stream
  python tests/benchmark_intrusion_comparison.py --source rtsp://admin:pass@192.168.1.100/stream1 --frames 300

The ROI defaults to a full-frame polygon; pass --roi to override.

Output: ASCII comparison table  (FPS, p50/p95 latency, CPU%, RSS MB)
"""

import sys
import os
import argparse
import time
import threading

# --- ensure project root is on PYTHONPATH -----------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    print("[bench] psutil not installed -- CPU/memory metrics disabled. "
          "Install with: pip install psutil")

# ============================================================================
# Helper: CPU sampler thread
# ============================================================================
class _CpuSampler(threading.Thread):
    """Samples cpu_percent and memory in a background thread while the pipeline runs."""
    def __init__(self):
        super().__init__(daemon=True)
        self.samples_cpu = []
        self.samples_mem = []
        self._stop = threading.Event()
        self._proc = psutil.Process() if _HAS_PSUTIL else None

    def run(self):
        while not self._stop.is_set():
            if self._proc:
                try:
                    self.samples_cpu.append(self._proc.cpu_percent(interval=None))
                    self.samples_mem.append(self._proc.memory_info().rss / 1e6)
                except Exception:
                    pass
            time.sleep(0.2)

    def stop(self):
        self._stop.set()

    def avg_cpu(self): return float(np.mean(self.samples_cpu)) if self.samples_cpu else 0.0
    def avg_mem(self): return float(np.mean(self.samples_mem)) if self.samples_mem else 0.0

# ============================================================================
# Runner: processes N frames from a pipeline and collects latency stats
# ============================================================================
def _run_pipeline(pipeline, source, roi_normalized, n_frames, label, dummy_output_dir):
    """
    Run pipeline.run_on_video() for 
_frames frames.
    Measures per-frame wall-clock latency including annotation.
    Returns dict of metrics.
    """
    print(f"\n  [{label}] Starting pipeline over {n_frames} frames ...")

    sampler = _CpuSampler()
    if _HAS_PSUTIL:
        sampler.start()

    latencies = []
    alert_count = 0
    frame_count = 0
    wall_start = time.perf_counter()

    try:
        gen = pipeline.run_on_video(
            input_path=source,
            output_dir=dummy_output_dir,
            roi_normalized=roi_normalized,
            config={},
        )
        for frame, alert_event in gen:
            frame_count += 1
            t_now = time.perf_counter()
            elapsed_ms = (t_now - wall_start) * 1000.0
            # Frame latency = elapsed / frame_count approximation
            if frame_count > 1:
                # Delta from last checkpoint
                latencies.append(elapsed_ms / frame_count)
            if alert_event:
                alert_count += 1
            if frame_count >= n_frames:
                break
    except StopIteration:
        pass
    except Exception as ex:
        print(f"  [{label}] ERROR during run: {ex}")

    wall_end = time.perf_counter()
    total_sec = wall_end - wall_start

    if _HAS_PSUTIL:
        sampler.stop()
        sampler.join(timeout=1.0)

    avg_fps   = frame_count / total_sec if total_sec > 0 else 0.0
    lats      = np.array(latencies) if latencies else np.array([0.0])
    p50_ms    = float(np.percentile(lats, 50))
    p95_ms    = float(np.percentile(lats, 95))
    cpu_pct   = sampler.avg_cpu() if _HAS_PSUTIL else -1
    mem_mb    = sampler.avg_mem() if _HAS_PSUTIL else -1

    return {
        "label":        label,
        "frames":       frame_count,
        "total_sec":    total_sec,
        "avg_fps":      avg_fps,
        "p50_ms":       p50_ms,
        "p95_ms":       p95_ms,
        "cpu_pct":      cpu_pct,
        "mem_mb":       mem_mb,
        "alert_count":  alert_count,
    }

# ============================================================================
# Pretty-print comparison table
# ============================================================================
def _print_table(results):
    cols = [
        ("Pipeline",          "label",       "%-30s"),
        ("Frames",            "frames",      "%6d"),
        ("Avg FPS",           "avg_fps",     "%8.2f"),
        ("p50 lat (ms)",      "p50_ms",      "%14.2f"),
        ("p95 lat (ms)",      "p95_ms",      "%14.2f"),
        ("CPU %",             "cpu_pct",     "%7.1f"),
        ("RAM (MB)",          "mem_mb",      "%10.1f"),
        ("Alerts fired",      "alert_count", "%13d"),
    ]
    header = "  ".join(f"{h:^{int(fmt[1:-1])}}" for h, _, fmt in cols)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  INTRUSION PIPELINE BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        row = "  ".join(fmt % r[key] for _, key, fmt in cols)
        print(row)
    print(sep)

    if len(results) == 2:
        base, new = results
        speedup = new["avg_fps"] / base["avg_fps"] if base["avg_fps"] > 0 else float("nan")
        print(f"\n  Speedup (new_intrusion / baseline): {speedup:.2f}x")
        if new["alert_count"] == base["alert_count"]:
            print(f"  Alert parity: PASS  ({new['alert_count']} alerts each)")
        else:
            print(f"  Alert parity: WARN — baseline={base['alert_count']}, "
                  f"new={new['alert_count']} (may differ by <patience window>)")
    print()

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark intrusion_detection vs new_intrusion pipeline."
    )
    parser.add_argument(
        "--source", default="0",
        help="Video file path, RTSP URL, or webcam index (default: 0)"
    )
    parser.add_argument(
        "--frames", type=int, default=300,
        help="Number of frames to process per pipeline (default: 300)"
    )
    parser.add_argument(
        "--roi", nargs=4, type=float,
        metavar=("X0 Y0 X1 Y1"),
        default=None,
        help="ROI as normalized corner coords e.g. 0.1 0.1 0.9 0.9"
    )
    parser.add_argument(
        "--pipeline", choices=["both", "baseline", "new"],
        default="both",
        help="Which pipeline(s) to benchmark (default: both)"
    )
    args = parser.parse_args()

    # Resolve ROI
    if args.roi:
        x0, y0, x1, y1 = args.roi
        roi = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    else:
        # Default: full-frame polygon (same as no-zone)
        roi = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    # Dummy output dir (alerts written here during benchmark)
    dummy_dir = os.path.join("storage", "alerts")
    os.makedirs(dummy_dir, exist_ok=True)

    print(f"\n  Source  : {args.source}")
    print(f"  Frames  : {args.frames}")
    print(f"  ROI     : {roi}")
    print(f"  Pipelines: {args.pipeline}")

    # ── Import both pipelines ────────────────────────────────────────────────
    from pipelines.intrusion_pipeline import IntrusionDetectionPipeline
    from pipelines.new_intrusion_pipeline import NewIntrusionPipeline

    results = []

    if args.pipeline in ("both", "baseline"):
        baseline = IntrusionDetectionPipeline()
        baseline.initialize()
        r = _run_pipeline(baseline, args.source, roi, args.frames,
                          "intrusion_detection (baseline)", dummy_dir)
        results.append(r)
        print(f"  [{r['label']}] done — {r['frames']} frames in {r['total_sec']:.1f}s "
              f"({r['avg_fps']:.1f} fps)")

    if args.pipeline in ("both", "new"):
        new_pipe = NewIntrusionPipeline()
        new_pipe.initialize()
        r = _run_pipeline(new_pipe, args.source, roi, args.frames,
                          "new_intrusion (OpenVINO)", dummy_dir)
        results.append(r)
        print(f"  [{r['label']}] done — {r['frames']} frames in {r['total_sec']:.1f}s "
              f"({r['avg_fps']:.1f} fps)")

    _print_table(results)

    # Exit code 0 = success
    sys.exit(0)


if __name__ == "__main__":
    main()
