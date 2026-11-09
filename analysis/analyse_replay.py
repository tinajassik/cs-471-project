"""Compare replayed real routing with uniform routing and across balancing policies.

Usage: python analysis/analyse_replay.py results/replay_*.csv

Reads benchmark CSVs from the replay job: trace rows (routing trace-<phase>-L<layer>)
and the uniform control at the same expert shape (routing zipf, alpha 0). For each
routing and batch size, prints the slowest GPU's median latency under each policy,
relative to uniform routing and to No Load Balance, with the token imbalance and the
most activated experts on one GPU. Needs only NumPy.
"""
import csv
import sys
from collections import defaultdict

import numpy as np


def step_latency(rows):
    """Slowest GPU's total per step, then the median over steps."""
    by_step = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(float(row["total_ms"]))
    return float(np.median([max(times) for times in by_step.values()]))


def load(paths):
    """Group rows by (world size, top-k, routing, balance, batch), skipping the per-rank split."""
    groups = defaultdict(list)
    for path in paths:
        with open(path, newline="") as file:
            for row in csv.DictReader(file):
                routing = row["routing"] if row["routing"] != "zipf" else f"zipf-a{float(row['alpha']):g}"
                key = (int(row["world_size"]), int(row["top_k"]), routing, row["balance"], int(row["batch_size"]))
                groups[key].append(row)
    return groups


def summarise(groups):
    summary = {}
    for key, rows in groups.items():
        gpu0 = [r for r in rows if r["rank"] == "0"]
        summary[key] = {
            "latency": step_latency(rows),
            "gpu_imbalance": float(np.median([float(r["gpu_imbalance"]) for r in gpu0])),
            "activated": float(np.median([float(r["max_activated_per_gpu"]) for r in gpu0])),
        }
    return summary


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    summary = summarise(load(paths))
    print("Latency in ms (slowest GPU, median over steps). vs uniform: the same policy's latency over uniform")
    print("routing at the same shape, No Load Balance. vs none: over No Load Balance with the same routing.")
    print(f"{'GPUs':>4} {'k':>2} {'routing':>20} {'balance':>8} {'batch':>6} | {'latency':>8} {'vs uniform':>10} "
          f"{'vs none':>8} | {'tok imb':>7} {'max act':>7}")
    for key in sorted(summary, key=lambda k: (k[0], k[1], k[2], k[4], k[3])):
        world_size, top_k, routing, balance, batch = key
        s = summary[key]
        uniform = summary.get((world_size, top_k, "zipf-a0", "none", batch))
        none = summary.get((world_size, top_k, routing, "none", batch))
        vs_uniform = s["latency"] / uniform["latency"] if uniform else float("nan")
        vs_none = s["latency"] / none["latency"] if none else float("nan")
        print(f"{world_size:4d} {top_k:2d} {routing:>20} {balance:>8} {batch:6d} | {s['latency']:8.3f} "
              f"{vs_uniform:10.2f} {vs_none:8.2f} | {s['gpu_imbalance']:7.2f} {s['activated']:7.0f}")


if __name__ == "__main__":
    main()
