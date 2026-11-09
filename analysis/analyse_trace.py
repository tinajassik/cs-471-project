"""Summarise a real model's routing and timing, recorded by moe_bench/model_trace.py.

Usage: python analysis/analyse_trace.py results/trace_qwen1.5.npz [--pick-layers]

For every MoE layer and phase: how skewed expert demand is, the Zipf alpha that best
matches it, and the GPU imbalance it causes under contiguous placement. Then whether
the hot experts stay the same over time, across workloads, and between prefill and
decode. If <trace>_timing.csv sits next to the trace, also what share of each forward
pass the MoE blocks take. Writes <trace>_layers.csv. Needs only NumPy.

--pick-layers prints only the least, median and most skewed layers (by prefill alpha),
comma-separated, for the replay job.
"""
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ALPHAS = np.round(np.arange(0, 3.0001, 0.02), 2)
FIT_SAMPLES = 20_000
POOR_FIT = 0.05  # total variation distance above which a Zipf curve is a poor description
LAYER_FIELDS = [
    "layer", "phase", "tokens", "hottest_share", "max_over_mean", "entropy", "gini",
    "alpha", "noise_alpha", "fit_distance", "gpu_imbalance_2", "gpu_imbalance_4",
]


def load(path):
    with np.load(path) as trace:
        data = {name: trace[name] for name in trace.files}
    data["meta"] = json.loads(str(data["meta"]))
    return data


def expert_counts(assignments, num_experts):
    return np.bincount(assignments.astype(np.int64).ravel(), minlength=num_experts).astype(float)


@lru_cache(maxsize=None)
def zipf_curves(num_experts, top_k):
    """Sorted assignment shares for each alpha, under the benchmark's sampling rule.

    A token picks top_k distinct experts with Zipf probabilities, which caps the hottest
    expert at 1/top_k of assignments. Sampling without replacement is the top-k of
    log p plus Gumbel noise, so it vectorises.
    """
    rng = np.random.default_rng(0)
    ranks = np.arange(1, num_experts + 1, dtype=float)
    curves = []
    for alpha in ALPHAS:
        log_p = -alpha * np.log(ranks)
        keys = log_p + rng.gumbel(size=(FIT_SAMPLES, num_experts))
        chosen = np.argpartition(-keys, top_k - 1, axis=1)[:, :top_k]
        counts = np.bincount(chosen.ravel(), minlength=num_experts)
        curves.append(np.sort(counts)[::-1] / counts.sum())
    return np.array(curves)


def fit_alpha(counts, top_k):
    """Return (alpha, total variation distance) of the closest Zipf curve to these counts."""
    shares = np.sort(counts)[::-1] / counts.sum()
    distances = 0.5 * np.abs(zipf_curves(len(counts), top_k) - shares).sum(axis=1)
    best = int(distances.argmin())
    return float(ALPHAS[best]), float(distances[best])


@lru_cache(maxsize=None)
def noise_alpha(num_tokens, num_experts, top_k):
    """The alpha a perfectly uniform router would fit with this many tokens, from sampling noise alone."""
    rng = np.random.default_rng(1)
    chosen = np.argpartition(rng.random((num_tokens, num_experts)), top_k - 1, axis=1)[:, :top_k]
    return fit_alpha(np.bincount(chosen.ravel(), minlength=num_experts).astype(float), top_k)[0]


def gini(counts):
    sorted_counts = np.sort(counts)
    n = len(counts)
    return float((2 * np.arange(1, n + 1) - n - 1) @ sorted_counts / (n * sorted_counts.sum()))


def normalised_entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def gpu_imbalance(counts, world_size):
    """Busiest GPU's assignments over the mean, with experts placed contiguously as in the benchmark."""
    owner = np.arange(len(counts)) * world_size // len(counts)
    per_gpu = np.bincount(owner, weights=counts, minlength=world_size)
    return float(per_gpu.max() / per_gpu.mean())


def ranks(values):
    """Ranks with ties averaged, for Spearman correlation."""
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values))
    result[order] = np.arange(len(values))
    for value in np.unique(values):
        tied = values == value
        result[tied] = result[tied].mean()
    return result


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def top_overlap(a, b, n=5):
    return len(set(np.argsort(a)[-n:]) & set(np.argsort(b)[-n:])) / n


def layer_rows(data):
    meta = data["meta"]
    rows = []
    for layer in data["layers"].tolist():
        for phase in ("prefill", "decode"):
            assignments = data[f"{phase}_{layer}"]
            if len(assignments) == 0:
                continue
            counts = expert_counts(assignments, meta["num_experts"])
            alpha, distance = fit_alpha(counts, meta["top_k"])
            rows.append({
                "layer": layer, "phase": phase, "tokens": len(assignments),
                "hottest_share": counts.max() / counts.sum(),
                "max_over_mean": counts.max() / counts.mean(),
                "entropy": normalised_entropy(counts),
                "gini": gini(counts),
                "alpha": alpha, "noise_alpha": noise_alpha(len(assignments), meta["num_experts"], meta["top_k"]),
                "fit_distance": distance,
                "gpu_imbalance_2": gpu_imbalance(counts, 2),
                "gpu_imbalance_4": gpu_imbalance(counts, 4),
            })
    return rows


def window_stability(assignments, prompt_of_token, num_experts, windows=8):
    """Mean Spearman correlation and top-5 overlap of expert loads between consecutive prompt windows."""
    edges = np.linspace(0, prompt_of_token.max() + 1, windows + 1)
    counts = [
        expert_counts(assignments[(prompt_of_token >= lo) & (prompt_of_token < hi)], num_experts)
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    counts = [c for c in counts if c.sum() > 0]
    pairs = list(zip(counts[:-1], counts[1:]))
    if not pairs:
        return float("nan"), float("nan")
    return (float(np.mean([spearman(a, b) for a, b in pairs])),
            float(np.mean([top_overlap(a, b) for a, b in pairs])))


def print_layers(rows, meta):
    print(f"\n{meta['model']}: {meta['num_experts']} experts, top-{meta['top_k']}")
    print("Hottest share: one expert's share of assignments (at most 1/top_k). Alpha: closest Zipf skew under")
    print(f"the benchmark's sampling rule, with its distance (> {POOR_FIT} is a poor fit). Noise: the alpha a uniform")
    print("router would fit with as many tokens. GPU imbalance: busiest GPU over the mean, contiguous placement.")
    print(f"{'layer':>5} {'phase':>7} {'tokens':>8} {'hottest':>8} {'max/mean':>8} {'entropy':>7} {'gini':>5} "
          f"{'alpha':>5} {'noise':>5} {'dist':>5} {'imb 2':>6} {'imb 4':>6}")
    for r in rows:
        flag = " poor fit" if r["fit_distance"] > POOR_FIT else ""
        print(f"{r['layer']:5d} {r['phase']:>7} {r['tokens']:8d} {r['hottest_share']:8.3f} {r['max_over_mean']:8.2f} "
              f"{r['entropy']:7.3f} {r['gini']:5.2f} {r['alpha']:5.2f} {r['noise_alpha']:5.2f} {r['fit_distance']:5.3f} "
              f"{r['gpu_imbalance_2']:6.2f} {r['gpu_imbalance_4']:6.2f}{flag}")
    for phase in ("prefill", "decode"):
        alphas = [r["alpha"] for r in rows if r["phase"] == phase]
        if alphas:
            print(f"{phase}: alpha {min(alphas):.2f} to {max(alphas):.2f} (median {np.median(alphas):.2f}); "
                  f"{sum(r['fit_distance'] > POOR_FIT for r in rows if r['phase'] == phase)} poor fits")


def print_stability(data):
    meta = data["meta"]
    num_experts = meta["num_experts"]
    sources = data["prompt_source"]
    print("\nSpearman correlation of expert loads (1 = the same experts are hot), per layer")
    print(f"{'layer':>5} {'prefill~decode':>14} {'windows':>8} {'top-5 kept':>10} " +
          " ".join(f"{'~'.join(pair):>18}" for pair in source_pairs(sources)))
    for layer in data["layers"].tolist():
        prefill, decode = data[f"prefill_{layer}"], data[f"decode_{layer}"]
        p_counts, d_counts = expert_counts(prefill, num_experts), expert_counts(decode, num_experts)
        across_phase = spearman(p_counts, d_counts) if d_counts.sum() else float("nan")
        windows, kept = window_stability(prefill, data["prefill_prompt"], num_experts)
        token_source = sources[data["prefill_prompt"]]
        by_source = " ".join(
            f"{spearman(expert_counts(prefill[token_source == a], num_experts), expert_counts(prefill[token_source == b], num_experts)):18.3f}"
            for a, b in source_pairs(sources)
        )
        print(f"{layer:5d} {across_phase:14.3f} {windows:8.3f} {kept:10.2f} {by_source}")


def source_pairs(sources):
    names = sorted(set(sources.tolist()))
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]


def print_timing(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    print(f"\nTime per forward pass on one GPU, from {path.name} (ms; MoE share is of total time)")
    print(f"{'phase':>8} {'tokens':>10} {'passes':>7} {'total':>8} {'attention':>9} {'moe':>8} {'other':>8} {'MoE share':>9}")
    bins = {"decode": [(1, 1)], "prefill": [(1, 127), (128, 511), (512, 10**9)]}
    for phase, ranges in bins.items():
        for lo, hi in ranges:
            group = [r for r in rows if r["phase"] == phase and lo <= int(r["tokens"]) <= hi]
            if not group:
                continue
            column = {c: np.array([float(r[c]) for r in group]) for c in ("total_ms", "attention_ms", "moe_ms", "other_ms")}
            label = "1" if hi == 1 else f"{lo}-{hi if hi < 10**9 else ''}"
            print(f"{phase:>8} {label:>10} {len(group):7d} {np.median(column['total_ms']):8.2f} "
                  f"{np.median(column['attention_ms']):9.2f} {np.median(column['moe_ms']):8.2f} "
                  f"{np.median(column['other_ms']):8.2f} {column['moe_ms'].sum() / column['total_ms'].sum():9.1%}")


def pick_layers(rows):
    prefill = sorted((r for r in rows if r["phase"] == "prefill"), key=lambda r: (r["alpha"], r["max_over_mean"]))
    picked = [prefill[0]["layer"], prefill[len(prefill) // 2]["layer"], prefill[-1]["layer"]]
    return ",".join(str(layer) for layer in dict.fromkeys(picked))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    path = Path(args[0])
    data = load(path)
    rows = layer_rows(data)
    if "--pick-layers" in sys.argv:
        print(pick_layers(rows))
        return
    print_layers(rows, data["meta"])
    with open(path.parent / f"{path.stem}_layers.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, LAYER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print_stability(data)
    timing = path.parent / f"{path.stem}_timing.csv"
    if timing.exists():
        print_timing(timing)
    print(f"\nwrote {path.parent / f'{path.stem}_layers.csv'}")


if __name__ == "__main__":
    main()
