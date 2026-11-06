"""Summarise benchmark CSVs: phases, Q3 predictors, cost fits and new token counts.

Usage: python analysis/analyse_results.py results/sweep_seed*.csv
Needs only NumPy. Older CSVs without new_sizes skip that section.
"""
import csv
import sys
from pathlib import Path

import numpy as np

FLOAT_COLUMNS = (
    "alpha", "batch_size", "step", "rank", "dispatch_ms", "compute_ms", "combine_ms", "total_ms",
    "tokens", "activated", "bytes_sent", "new_sizes", "max_tokens_per_gpu", "max_activated_per_gpu",
)


def load(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    data = {c: np.array([float(r[c]) for r in rows]) for c in FLOAT_COLUMNS if c in rows[0]}
    data["info"] = {c: rows[0].get(c, "?") for c in ("seed", "expert_size", "layers", "warm_passes", "balance")}
    return data


def fit(predictors, y):
    """Least squares with an intercept. Returns (coefficients, R squared)."""
    X = np.column_stack([np.ones(len(y)), *predictors])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = ((y - X @ beta) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return beta, (1 - residual / total if total > 0 else float("nan"))


def correlation(x, y):
    return float("nan") if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])


def per_step(data, mask):
    """Return (slowest GPU's total, slowest GPU's compute, routing columns) for each step."""
    ranks = np.unique(data["rank"][mask])
    gpu0 = mask & (data["rank"] == ranks[0])

    def slowest(column):
        return np.max([data[column][mask & (data["rank"] == r)] for r in ranks], axis=0)

    return (
        slowest("total_ms"),
        slowest("compute_ms"),
        data["max_tokens_per_gpu"][gpu0],
        data["max_activated_per_gpu"][gpu0],
    )


def print_phases(data):
    ranks = np.unique(data["rank"]).astype(int)
    gpu_headers = " | ".join(f"{f'g{r} disp':>7} {f'g{r} comp':>8} {f'g{r} comb':>8}" for r in ranks)
    print(f"{'alpha':>5} {'batch':>6} | {'latency':>8} {'p90':>8} {'tokens/s':>12} | {gpu_headers}")
    for alpha in np.unique(data["alpha"]):
        for batch in np.unique(data["batch_size"]):
            mask = (data["alpha"] == alpha) & (data["batch_size"] == batch)
            if not mask.any():
                continue
            latency = per_step(data, mask)[0]
            gpu_medians = " | ".join(
                " ".join(
                    f"{np.median(data[f'{phase}_ms'][mask & (data['rank'] == rank)]):{width}.3f}"
                    for phase, width in (("dispatch", 7), ("compute", 8), ("combine", 8))
                )
                for rank in ranks
            )
            print(f"{alpha:5.1f} {int(batch):6d} | {np.median(latency):8.3f} {np.percentile(latency, 90):8.3f} "
                  f"{batch / np.median(latency) * 1000:12,.0f} | {gpu_medians}")


def print_q3(data):
    print(f"{'batch':>6} | {'compute~max tokens':>18} {'compute~max activated':>21} | "
          f"{'latency~max tokens':>18} {'latency~max activated':>21}")
    for batch in np.unique(data["batch_size"]):
        latency, compute, tokens, activated = per_step(data, data["batch_size"] == batch)
        print(f"{int(batch):6d} | {correlation(tokens, compute):18.3f} {correlation(activated, compute):21.3f} | "
              f"{correlation(tokens, latency):18.3f} {correlation(activated, latency):21.3f}")


def cost_fit(data, mask):
    """Fit compute = c + a * activated + b * tokens. Returns (c, a, b, R^2) with costs in us.

    When activated barely varies, as at large batches, a can't be separated from c,
    so it's returned as NaN and only tokens are fitted.
    """
    activated, tokens, compute = data["activated"][mask], data["tokens"][mask], data["compute_ms"][mask]
    if np.ptp(activated) < 0.25 * activated.max():
        beta, r2 = fit([tokens], compute)
        return beta[0] * 1000, float("nan"), beta[1] * 1000, r2
    beta, r2 = fit([activated, tokens], compute)
    return beta[0] * 1000, beta[1] * 1000, beta[2] * 1000, r2


def print_cost_fits(data):
    batch = data["batch_size"]
    for label, mask in (("batch <= 64", batch <= 64), ("batch <= 256", batch <= 256),
                        ("batch >= 1024", batch >= 1024), ("batch >= 16384", batch >= 16384)):
        if mask.sum() < 3:
            continue
        c, per_expert, per_token, r2 = cost_fit(data, mask)
        print(f"{label:>15}: constant {c:9.1f} us, per expert {per_expert:8.2f} us, "
              f"per token {per_token:8.4f} us, R^2 {r2:.3f}")


def print_new_sizes(data):
    print(f"Mean new token counts per GPU-step: {data['new_sizes'].mean():.3f}")
    large = data["batch_size"] >= 1024
    for count in range(int(data["new_sizes"][large].max()) + 1 if large.any() else 0):
        mask = large & (data["new_sizes"] == count)
        if mask.any():
            print(f"  {count} new: {mask.sum():5d} GPU-steps, median compute {np.median(data['compute_ms'][mask]):.3f} ms")


def print_across_files(named):
    print(f"{'file':>42} {'seed':>5} {'size':>5} {'layers':>6} {'warm':>4} {'balance':>8} | "
          f"{'per expert us':>13} {'constant us':>11} {'R^2':>6} {'new/step':>8} | {'per token us (batch >= 1024)':>28}")
    for name, data in named:
        info = data["info"]
        small = data["batch_size"] <= 64
        large = data["batch_size"] >= 1024
        c, per_expert, _, r2 = cost_fit(data, small) if small.sum() >= 3 else (*[float("nan")] * 3, float("nan"))
        per_token = cost_fit(data, large)[2] if large.sum() >= 3 else float("nan")
        new = data["new_sizes"].mean() if "new_sizes" in data else float("nan")
        print(f"{name:>42} {info['seed']:>5} {info['expert_size']:>5} {info['layers']:>6} {info['warm_passes']:>4} "
              f"{info['balance']:>8} | "
              f"{per_expert:13.2f} {c:11.1f} {r2:6.3f} {new:8.3f} | {per_token:28.4f}")


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    named = [(Path(path).name, load(path)) for path in paths]
    for name, data in named:
        print(f"\n===== {name} =====")
        print("\nMedian times in ms. Latency is the slowest GPU in each step.")
        print_phases(data)
        print("\nQ3: correlation with each load measure, across skews at one batch size")
        print_q3(data)
        print("\nCost fits per GPU per step")
        print_cost_fits(data)
        if "new_sizes" in data:
            print("\nNever-seen expert token counts in timed steps (batch >= 1024)")
            print_new_sizes(data)
    print("\n===== Across files (fit on batch <= 64) =====")
    print_across_files(named)


if __name__ == "__main__":
    main()
