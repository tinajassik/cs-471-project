"""Show whether the compute phase was waiting on the CPU or on the GPU.

Usage: python analysis/analyse_timeline.py results/timeline_*.sqlite
       python analysis/analyse_timeline.py --range moe --last 0 results/model_timeline_*.sqlite
Reads the files made by `nsys export --type=sqlite`. Needs only NumPy. --range picks the
NVTX ranges to read: compute (the benchmark's phase, the default), or moe or attention
(model_trace.py's blocks, named "moe 3" and so on). --last N reads each process's last
N ranges, and 0 reads all of them.

Low GPU busy, with launch gaps much longer than kernels, means the GPU sat idle
waiting for the CPU. Busy near 100%, with the GPU finishing after the CPU, means
the CPU was waiting for the GPU.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np

# A globalTid holds the process ID above this bit, and a globalPid holds only that part.
PROCESS_SHIFT = 0x1000000
NVTX_PUSH_POP_RANGE = 59

COMPUTE_PHASES = f"""
    SELECT n.globalTid, n.start, n."end"
    FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
    WHERE n.eventType = {NVTX_PUSH_POP_RANGE} AND n."end" IS NOT NULL
        AND (COALESCE(n.text, s.value) = ?1 OR COALESCE(n.text, s.value) LIKE ?1 || ' %')
    ORDER BY n.start
"""
# Correlation IDs repeat between processes, so the join has to match the process too.
KERNELS = f"""
    SELECT r.globalTid, r.start, k.start, k."end"
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
        ON k.correlationId = r.correlationId AND k.globalPid / {PROCESS_SHIFT} = r.globalTid / {PROCESS_SHIFT}
    ORDER BY r.start
"""


def busy_time(intervals):
    """Return the time covered by at least one (start, end) interval."""
    busy, covered_until = 0, None
    for start, end in sorted(intervals):
        if covered_until is None or start > covered_until:
            busy += end - start
            covered_until = end
        elif end > covered_until:
            busy += end - covered_until
            covered_until = end
    return busy


def phase_stats(phase_start, phase_end, kernels):
    """Describe one compute phase, from the kernels the CPU launched during it.

    kernels holds (launch time, kernel start, kernel end), sorted by launch time and in
    ns. Returns None if the phase launched none. Results are in us.
    """
    launched = [k for k in kernels if phase_start <= k[0] < phase_end]
    if not launched:
        return None
    runs = [(start, end) for _, start, end in launched]
    gpu_start, gpu_end = min(r[0] for r in runs), max(r[1] for r in runs)
    launches = [k[0] for k in launched]
    return {
        "cpu_us": (phase_end - phase_start) / 1000,
        "span_us": (gpu_end - gpu_start) / 1000,
        "busy_us": busy_time(runs) / 1000,
        "gpu_after_cpu_us": (gpu_end - phase_end) / 1000,
        "kernels": len(launched),
        "launch_gap_us": float(np.median(np.diff(launches))) / 1000 if len(launches) > 1 else float("nan"),
        "kernel_us": float(np.median([end - start for start, end in runs])) / 1000,
    }


def summarise(path, last=20, range_name="compute"):
    """Return the median stats per process over its last compute phases, the timed steps."""
    # Read-only, because otherwise SQLite creates an empty database for a missing file.
    with sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True) as connection:
        phases = connection.execute(COMPUTE_PHASES, (range_name,)).fetchall()
        kernels = connection.execute(KERNELS).fetchall()
    summaries = {}
    for thread in sorted({phase[0] for phase in phases}):
        own_kernels = [k[1:] for k in kernels if k[0] == thread]
        own_phases = [p[1:] for p in phases if p[0] == thread][-last:]  # [-0:] keeps them all
        stats = [s for s in (phase_stats(*p, own_kernels) for p in own_phases) if s]
        if stats:
            medians = {key: float(np.median([s[key] for s in stats])) for key in stats[0]}
            summaries[thread // PROCESS_SHIFT % PROCESS_SHIFT] = {**medians, "phases": len(stats)}
    return summaries


def print_summaries(named):
    print(f"{'file':>38} {'process':>8} {'phases':>6} | {'CPU issue ms':>12} {'GPU span ms':>11} {'GPU busy ms':>11} "
          f"{'busy %':>6} {'GPU after CPU us':>16} | {'kernels':>7} {'launch gap us':>13} {'kernel us':>9}")
    for name, summaries in named:
        for process, s in summaries.items():
            print(f"{name:>38} {process:>8} {s['phases']:>6} | {s['cpu_us'] / 1000:12.3f} {s['span_us'] / 1000:11.3f} "
                  f"{s['busy_us'] / 1000:11.3f} {100 * s['busy_us'] / s['span_us']:6.1f} {s['gpu_after_cpu_us']:16.1f} | "
                  f"{s['kernels']:7.0f} {s['launch_gap_us']:13.1f} {s['kernel_us']:9.1f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("--range", default="compute")
    parser.add_argument("--last", type=int, default=20)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    print_summaries([(Path(path).name, summarise(path, args.last, args.range)) for path in args.paths])


if __name__ == "__main__":
    main()
