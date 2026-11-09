"""Make figures and tables from the benchmark results.

Usage: python analysis/report_figures.py [results dir] [output dir]
Defaults are results/ and figures/. Needs pandas and matplotlib.

Latency is per step: the slowest GPU's total, then the median over steps. Seeds
change which experts are hot, so figures show the mean over seeds and their range.
"""
import sqlite3
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt 

import analyse_timeline  

# Kuma's GPUs are the 94 GB H100 SXM, whose peaks we haven't measured, so derived rates
# are reported as absolute numbers rather than as a share of a datasheet peak.
HIDDEN = 2048

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
# Categorical slots, validated as a set (CVD worst pair dE 9.2).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
POLICY_COLOR = {"none": BLUE, "eplb-1.0": ORANGE, "eplb-1.5": AQUA}
POLICY_MARKER = {"none": "o", "eplb-1.0": "s", "eplb-1.5": "^"}
POLICY_LABEL = {"none": "No Load Balance", "eplb-1.0": "EPLB 1.0x", "eplb-1.5": "EPLB 1.5x"}
# Alpha and expert size are magnitudes, so they get one blue ramp, light to dark.
RAMP = ["#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.5, "lines.markersize": 4,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
SINGLE = (3.5, 2.3) 
DOUBLE = (7.16, 2.4)


def load(results, pattern):
    frames = [pd.read_csv(path).assign(file=path.name) for path in sorted(results.glob(pattern))]
    if not frames:
        sys.exit(f"no files match {results / pattern}")
    data = pd.concat(frames, ignore_index=True)
    if "balance" not in data:
        data["balance"] = "none"
    return data


def per_config(data, keys):
    """Median over steps of the slowest GPU's total and compute, per configuration."""
    steps = data.groupby([*keys, "step"]).agg(latency=("total_ms", "max"), compute=("compute_ms", "max"))
    return steps.groupby(keys).median().reset_index()


def across_seeds(table, keys, column):
    return table.groupby(keys)[column].agg(["mean", "min", "max"]).reset_index()


def log2_batches(ax, batches, rotation=0):
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([f"{b // 1024}K" if b >= 1024 else str(b) for b in batches], rotation=rotation)
    ax.minorticks_off()


def save(fig, out, name):
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def assignment_share(alpha, num_experts=60, top_k=4, tokens=100_000, seed=0):
    """Share of assignments per popularity rank, with experts distinct within a token.

    Taking the top k of log p plus Gumbel noise samples k without replacement, as
    torch.multinomial does in the benchmark, so the hottest experts are capped the same way.
    """
    p = np.arange(1, num_experts + 1, dtype=float) ** -alpha
    p /= p.sum()
    noise = np.random.default_rng(seed).gumbel(size=(tokens, num_experts))
    chosen = np.argpartition(-(np.log(p) + noise), top_k, axis=1)[:, :top_k]
    return np.bincount(chosen.ravel(), minlength=num_experts) / chosen.size


def q1_setup(results, out):
    """What the baseline looks like: demand per expert, and the split it gives each GPU."""
    sweep = load(results, "sweep_seed*.csv")
    largest = sweep[sweep.batch_size == sweep.batch_size.max()]
    tokens = largest.groupby(["seed", "alpha", "rank"]).tokens.median().unstack()
    hot_share = (tokens.max(axis=1) / tokens.sum(axis=1)).reset_index(name="share")

    fig, (left, right) = plt.subplots(1, 2, figsize=DOUBLE)
    ranks = np.arange(1, 61)
    for i, alpha in enumerate(sorted(sweep.alpha.unique())):
        left.plot(ranks, 100 * assignment_share(alpha), color=RAMP[i], marker=MARKERS[i], markevery=6,
                  label=f"α = {alpha:g}")
    left.set_yscale("log")
    left.set_yticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
    left.minorticks_off()
    left.set_xlabel("Expert, by popularity rank")
    left.set_ylabel("Share of token assignments (%)")
    left.set_title("(a) Demand per expert (uniform share: 1.7%)", loc="left")
    left.legend(loc="upper right")

    for color, marker, seed in zip((BLUE, ORANGE, AQUA), "os^", sorted(hot_share.seed.unique())):
        rows = hot_share[hot_share.seed == seed]
        right.plot(rows.alpha, 100 * rows.share, color=color, marker=marker, label=f"seed {seed}")
    right.axhline(50, color=INK_2, linewidth=0.8)
    right.set_xticks(sorted(sweep.alpha.unique()))
    right.set_xlabel("Zipf skew α")
    right.set_ylabel("Busier GPU's share of assignments (%)")
    right.set_title("(b) Resulting split across 2 GPUs (contiguous)", loc="left")
    right.legend(loc="upper left")
    save(fig, out, "q1_setup")
    return hot_share


def q1_utilization(results, out):
    """Share of each step a GPU spends computing: the busier and the idler GPU, uniform against skewed."""
    data = load(results, "sweep_seed*.csv")
    step_latency = data.groupby(["seed", "alpha", "batch_size", "step"]).total_ms.transform("max")
    data = data.assign(busy=data.compute_ms / step_latency)
    per_gpu = data.groupby(["seed", "alpha", "batch_size", "rank"]).busy.median().reset_index()
    group = per_gpu.groupby(["seed", "alpha", "batch_size"]).busy
    per_gpu["role"] = np.where(per_gpu.busy == group.transform("max"), "busier", "idler")
    shares = per_gpu.groupby(["alpha", "batch_size", "role"]).busy.mean().unstack().reset_index()

    fig, ax = plt.subplots(figsize=SINGLE)
    uniform = shares[shares.alpha == 0]
    ax.plot(uniform.batch_size, 100 * (uniform.busier + uniform.idler) / 2, color=INK_2, linestyle="--",
            marker="o", markerfacecolor="white", label="α = 0, either GPU")
    skewed = shares[shares.alpha == 1.5]
    ax.plot(skewed.batch_size, 100 * skewed.busier, color=ORANGE, marker="^", label="α = 1.5, busier GPU")
    ax.plot(skewed.batch_size, 100 * skewed.idler, color=BLUE, marker="s", label="α = 1.5, idler GPU")
    log2_batches(ax, sorted(shares.batch_size.unique()))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Batch size (tokens per step, all GPUs)")
    ax.set_ylabel("Compute time / step latency (%)")
    ax.legend(loc="lower center")
    save(fig, out, "q1_utilization")
    return shares


def q1_skew(results, out):
    """Latency against batch per alpha, and relative to uniform routing."""
    sweep = per_config(load(results, "sweep_seed*.csv"), ["seed", "alpha", "batch_size"])
    uniform = sweep[sweep.alpha == 0].set_index(["seed", "batch_size"]).latency
    sweep["relative"] = sweep.latency / uniform.loc[list(zip(sweep.seed, sweep.batch_size))].to_numpy()
    batches = sorted(sweep.batch_size.unique())

    fig, (left, right) = plt.subplots(1, 2, figsize=DOUBLE)
    for i, alpha in enumerate(sorted(sweep.alpha.unique())):
        rows = sweep[sweep.alpha == alpha]
        style = dict(color=RAMP[i], marker=MARKERS[i], label=f"α = {alpha:g}")
        lat = across_seeds(rows, ["batch_size"], "latency")
        left.plot(lat.batch_size, lat["mean"], **style)
        rel = across_seeds(rows, ["batch_size"], "relative")
        right.plot(rel.batch_size, rel["mean"], **style)
        right.fill_between(rel.batch_size, rel["min"], rel["max"], color=RAMP[i], alpha=0.15, linewidth=0)
    left.set_yscale("log")
    left.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    left.minorticks_off()
    left.set_ylabel("MoE-layer latency (ms)")
    left.set_title("(a) Latency, 2 GPUs, No Load Balance", loc="left")
    right.axhline(1, color=INK_2, linewidth=0.8)
    right.set_ylabel("Latency relative to α = 0")
    right.set_title("(b) Effect of skew (band: range over 3 seeds)", loc="left")
    for ax in (left, right):
        log2_batches(ax, batches)
        ax.set_xlabel("Batch size (tokens per step, all GPUs)")
    left.legend(loc="upper left")
    save(fig, out, "q1_latency")
    return sweep


def q1_straggler(results, out):
    """Per-GPU phase times at the largest batch, uniform against skewed."""
    data = load(results, "sweep_seed471_*.csv")
    data = data[(data.batch_size == data.batch_size.max()) & data.alpha.isin([0, 1.5])]
    phases = data.groupby(["alpha", "rank"])[["dispatch_ms", "compute_ms", "combine_ms"]].median()

    fig, ax = plt.subplots(figsize=(SINGLE[0], 1.7))
    labels = [f"α = {alpha:g}, GPU {rank}" for alpha, rank in phases.index]
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for column, color, name in (("dispatch_ms", BLUE, "dispatch"), ("compute_ms", ORANGE, "compute"),
                                ("combine_ms", AQUA, "combine (incl. waiting)")):
        ax.barh(y, phases[column], left=left, color=color, height=0.6, label=name,
                edgecolor="white", linewidth=1)
        left += phases[column].to_numpy()
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Median time per step (ms), batch {int(data.batch_size.max()):,}, seed 471")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, handlelength=1)
    save(fig, out, "q1_straggler")
    return phases


def q2_eplb(results, out, alpha=1.5):
    """EPLB latency relative to No Load Balance, on 2 and 4 GPUs."""
    table = per_config(load(results, "eplb_*seed*.csv"), ["world_size", "balance", "seed", "alpha", "batch_size"])
    base = table[table.balance == "none"].set_index(["world_size", "seed", "alpha", "batch_size"]).latency
    keys = list(zip(table.world_size, table.seed, table.alpha, table.batch_size))
    table["ratio"] = table.latency / base.loc[keys].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE, sharey=True)
    for ax, world_size in zip(axes, (2, 4)):
        rows = table[(table.world_size == world_size) & (table.alpha == alpha)]
        batches = sorted(rows.batch_size.unique())
        if world_size == 4:
            # Dispatch stalls from CPU thread contention inflate these batches by 10-18%.
            ax.axvspan(768, 24576, color=GRID, alpha=0.6, linewidth=0)
            ax.text(4096, 1.72, "CPU contention\n(re-run planned)", ha="center", va="top", fontsize=6, color=INK_2)
        ax.axhline(1, color=INK_2, linewidth=0.8)
        for balance in ("eplb-1.0", "eplb-1.5"):
            ratio = across_seeds(rows[rows.balance == balance], ["batch_size"], "ratio")
            ax.plot(ratio.batch_size, ratio["mean"], color=POLICY_COLOR[balance], marker=POLICY_MARKER[balance],
                    label=POLICY_LABEL[balance])
            ax.fill_between(ratio.batch_size, ratio["min"], ratio["max"], color=POLICY_COLOR[balance],
                            alpha=0.15, linewidth=0)
        log2_batches(ax, batches, rotation=45)
        ax.set_xlabel("Batch size (tokens per step, all GPUs)")
        ax.set_title(f"({'ab'[world_size // 4]}) {world_size} GPUs, α = {alpha:g}", loc="left")
    axes[0].set_ylabel("Latency relative to No Load Balance")
    axes[0].legend(loc="lower left")
    save(fig, out, "q2_eplb")
    return table


def q2_breakdown(results, out, alpha=1.5):
    """Per-GPU phase times for each policy at the largest 2-GPU batch, seed 471."""
    data = load(results, "eplb_[ne]*seed471_*.csv")
    data = data[(data.alpha == alpha) & (data.batch_size == data.batch_size.max())]
    phases = data.groupby(["balance", "rank"])[["dispatch_ms", "compute_ms", "combine_ms"]].median()
    phases = phases.reindex(list(POLICY_LABEL), level="balance")

    fig, ax = plt.subplots(figsize=(SINGLE[0], 2.1))
    labels = [f"{POLICY_LABEL[balance]}, GPU {rank}" for balance, rank in phases.index]
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for column, color, name in (("dispatch_ms", BLUE, "dispatch"), ("compute_ms", ORANGE, "compute"),
                                ("combine_ms", AQUA, "combine (incl. waiting)")):
        ax.barh(y, phases[column], left=left, color=color, height=0.6, label=name, edgecolor="white", linewidth=1)
        left += phases[column].to_numpy()
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Median time per step (ms), batch {int(data.batch_size.max()) // 1024}K, α = {alpha:g}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, handlelength=1)
    save(fig, out, "q2_breakdown")
    return phases


def q2_activated(results, out, alpha=1.5):
    """Most activated experts on one GPU per step, for each policy, on 2 and 4 GPUs."""
    data = load(results, "eplb_*seed*.csv")
    data = data[(data.alpha == alpha) & (data["rank"] == 0)]  # the max is the same on every rank's row
    activated = (data.groupby(["world_size", "balance", "seed", "batch_size"]).max_activated_per_gpu.median()
                 .groupby(["world_size", "balance", "batch_size"]).mean().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE, sharey=True)
    for ax, world_size in zip(axes, (2, 4)):
        rows = activated[activated.world_size == world_size]
        for balance in POLICY_LABEL:
            mine = rows[rows.balance == balance]
            ax.plot(mine.batch_size, mine.max_activated_per_gpu, color=POLICY_COLOR[balance],
                    marker=POLICY_MARKER[balance], label=POLICY_LABEL[balance])
        log2_batches(ax, sorted(rows.batch_size.unique()), rotation=45)
        ax.set_xlabel("Batch size (tokens per step, all GPUs)")
        ax.set_title(f"({'ab'[world_size // 4]}) {world_size} GPUs, α = {alpha:g}", loc="left")
    axes[0].set_ylabel("Most activated experts on one GPU")
    axes[0].legend(loc="upper left")
    save(fig, out, "q2_activated")
    return activated


PREDICTORS = {"max_tokens_per_gpu": ("Max tokens per GPU", BLUE, "o"),
              "max_activated_per_gpu": ("Max activated experts per GPU", ORANGE, "s")}


def q3_configs(results):
    """One row per (GPUs, policy, seed, alpha, batch): medians over steps of latency and both load measures.

    The three policies decouple the measures: EPLB 1.0x lowers max tokens at the same activated
    count, and 1.5x raises activated experts at a similar token load.
    """
    data = load(results, "eplb_*seed*.csv")
    keys = ["world_size", "balance", "seed", "alpha", "batch_size"]
    steps = data.groupby([*keys, "step"]).agg(latency=("total_ms", "max"), **{
        column: (column, "first") for column in PREDICTORS})  # the max measures are the same on every rank
    return steps.groupby(keys).median().reset_index()


def r_squared(x, y):
    return float("nan") if np.std(x) == 0 or np.std(y) == 0 else float(np.corrcoef(x, y)[0, 1] ** 2)


def q3_r2(configs, out):
    """R^2 of a straight-line fit of latency on each load measure, per batch size, across 18 configurations."""
    rows = []
    for (world_size, batch), group in configs.groupby(["world_size", "batch_size"]):
        rows.append({"world_size": world_size, "batch_size": batch, "configs": len(group),
                     **{column: r_squared(group[column], group.latency) for column in PREDICTORS}})
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE, sharey=True)
    for ax, world_size in zip(axes, (2, 4)):
        rows = table[table.world_size == world_size]
        if world_size == 4:
            ax.axvspan(768, 24576, color=GRID, alpha=0.6, linewidth=0)
            ax.text(4096, 0.5, "CPU contention\n(re-run planned)", ha="center", va="center", fontsize=6, color=INK_2)
        for column, (label, color, marker) in PREDICTORS.items():
            ax.plot(rows.batch_size, rows[column], color=color, marker=marker, label=label)
        log2_batches(ax, sorted(rows.batch_size.unique()), rotation=45)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Batch size (tokens per step, all GPUs)")
        ax.set_title(f"({'ab'[world_size // 4]}) {world_size} GPUs", loc="left")
    axes[0].set_ylabel("R² of layer latency")
    axes[0].legend(loc="center left")
    save(fig, out, "q3_r2")
    return table


def expert_size_control(results, out):
    """Compute against activated experts with 1 to 1/64 of the weight bytes."""
    data = load(results, "control_size*.csv")
    data = data[data.batch_size <= 64]
    per_gpu = (data.groupby(["expert_size", "alpha", "batch_size", "rank"])
               [["compute_ms", "activated"]].median().reset_index())

    fig, ax = plt.subplots(figsize=SINGLE)
    for i, size in enumerate(sorted(per_gpu.expert_size.unique(), reverse=True)):
        rows = per_gpu[per_gpu.expert_size == size].sort_values("activated")
        megabytes = 3 * HIDDEN * size * 2 / 1e6
        # Hollow and shrinking, so points from every size stay visible where they coincide.
        ax.plot(rows.activated, rows.compute_ms, color=RAMP[i], marker=MARKERS[i], linestyle="none",
                markersize=8 - 1.5 * i, markerfacecolor="none", markeredgewidth=1.2,
                label=f"{megabytes:.2f} MB per expert" if megabytes < 1 else f"{megabytes:.1f} MB per expert")
    ax.set_xlabel("Activated experts on the GPU (batches 1 to 64)")
    ax.set_ylabel("Compute time (ms)")
    ax.legend(loc="upper left")
    save(fig, out, "method_expert_size")


def stalled(dispatch):
    """A step whose slowest dispatch is over 3x the configuration's median, plus 0.2 ms for tiny batches."""
    return dispatch > 3 * dispatch.median() + 0.2


def contention_tables(results):
    """The evidence for CPU thread contention on 4 GPUs, as three tables.

    1. Each diagnostic run (job diag_4gpu): the slowest GPU's dispatch per step, and latency.
    2. Share of stalled steps per batch, in the 4-GPU sweep against the 2-GPU sweeps.
    3. The same configuration in the 4-GPU sweep against the diagnostic runs, to size the inflation.
    """
    diag = load(results, "diag_*.csv")
    diag["run"] = diag.file.str.extract(r"diag_(.+)_\d+\.csv")[0]
    steps = diag.groupby(["run", "batch_size", "step"]).agg(
        dispatch=("dispatch_ms", "max"), latency=("total_ms", "max"), gpus=("world_size", "first"))
    runs = steps.groupby(["run", "batch_size"]).agg(
        gpus=("gpus", "first"), dispatch_median=("dispatch", "median"),
        dispatch_p90=("dispatch", lambda d: d.quantile(0.9)), dispatch_max=("dispatch", "max"),
        latency_median=("latency", "median")).reset_index()

    shares = []
    for name, pattern in (("4 GPUs, EPLB sweep", "eplb_4gpu_*seed*.csv"), ("2 GPUs, EPLB sweep", "eplb_[ne]*seed*.csv"),
                          ("2 GPUs, Q1 sweep", "sweep_seed*.csv")):
        data = load(results, pattern)
        steps = data.groupby(["file", "alpha", "batch_size", "step"]).dispatch_ms.max()
        flags = steps.groupby(["file", "alpha", "batch_size"], group_keys=False).apply(stalled)
        share = flags.groupby("batch_size").mean().rename(name)
        shares.append(share)
    shares = (100 * pd.concat(shares, axis=1)).sort_index().reset_index()

    # The diagnostic ran seed 471, No Load Balance, alpha 1.5: the same configuration as this sweep file.
    sweep = per_config(load(results, "eplb_4gpu_none_seed471_*.csv").query("alpha == 1.5"), ["batch_size"])
    medians = runs[runs.run.isin(["all4", "all4_1thread"])].pivot(index="batch_size", columns="run",
                                                                   values="latency_median")
    inflation = sweep.set_index("batch_size")[["latency"]].rename(columns={"latency": "sweep"}).join(medians, how="inner")
    inflation["sweep / 1 thread"] = inflation.sweep / inflation.all4_1thread
    return runs, shares, inflation.reset_index()


GEMM_KERNELS = f"""
    SELECT r.globalTid, r.start, k."end" - k.start, s.value
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
        ON k.correlationId = r.correlationId
        AND k.globalPid / {analyse_timeline.PROCESS_SHIFT} = r.globalTid / {analyse_timeline.PROCESS_SHIFT}
    JOIN StringIds s ON k.shortName = s.id
    WHERE s.value LIKE 'nvjet%' OR s.value LIKE '%gemm%'
"""


def gemm_rates(sqlite_path, last=20):
    """Return median GEMM time per compute phase (us) and the GEMM kernels' mean grid size.

    Tiny experts use generic cuBLAS kernels (Kernel2, splitKreduce) instead, so they return NaN.
    """
    uri = f"{sqlite_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        phases = connection.execute(analyse_timeline.COMPUTE_PHASES).fetchall()
        kernels = connection.execute(GEMM_KERNELS).fetchall()
        blocks = connection.execute(
            "SELECT AVG(k.gridX * k.gridY * k.gridZ) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName = s.id WHERE s.value LIKE 'nvjet%' OR s.value LIKE '%gemm%'"
        ).fetchone()[0]
    if blocks is None:
        return float("nan"), float("nan")
    per_phase = []
    for thread in {phase[0] for phase in phases}:
        for start, end in [p[1:] for p in phases if p[0] == thread][-last:]:
            per_phase.append(sum(k[2] for k in kernels if k[0] == thread and start <= k[1] < end) / 1000)
    return float(np.median(per_phase)), float(blocks)


def timeline_table(results):
    """Rows for the GPU busy and derived bandwidth table, from the Nsight Systems traces."""
    rows = []
    for path in sorted(results.glob("timeline_*.sqlite")):
        run = pd.read_csv(path.with_suffix(".csv"))
        size, batch = int(run.expert_size[0]), int(run.batch_size[0])
        busy = analyse_timeline.summarise(path)
        gemm_us, blocks = gemm_rates(path)
        tokens, activated = run.tokens.median(), run.activated.median()
        weight_bytes = activated * 3 * HIDDEN * size * 2
        flops = 2 * 3 * HIDDEN * size * tokens
        rows.append({
            "batch": batch, "expert size": size,
            "GPU busy %": np.median([100 * s["busy_us"] / s["span_us"] for s in busy.values()]),
            "launch gap us": np.median([s["launch_gap_us"] for s in busy.values()]),
            "GEMM blocks (of 132 SMs)": blocks,
            "GEMM TB/s (weights)": weight_bytes / gemm_us / 1e6,
            "GEMM TFLOP/s": flops / gemm_us / 1e6,
        })
    return pd.DataFrame(rows).sort_values(["batch", "expert size"], ascending=[True, False])


def markdown(frame, floatfmt=".2f"):
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    lines = [header, "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False):
        cells = [f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
    out.mkdir(parents=True, exist_ok=True)

    hot_share = q1_setup(results, out)
    sweep = q1_skew(results, out)
    shares = q1_utilization(results, out)
    phases = q1_straggler(results, out)
    eplb = q2_eplb(results, out)
    q2_phases = q2_breakdown(results, out)
    q2_activated(results, out)
    configs = q3_configs(results)
    r2 = q3_r2(configs, out)
    expert_size_control(results, out)
    timeline = timeline_table(results)
    diag_runs, stall_shares, inflation = contention_tables(results)

    q1 = sweep.groupby(["batch_size", "alpha"]).latency.mean().unstack()
    q1.columns = [f"α={a:g} (ms)" for a in q1.columns]
    q1["α=1.5 vs α=0"] = q1.iloc[:, -1] / q1.iloc[:, 0]
    q2 = (eplb[eplb.balance != "none"].groupby(["world_size", "alpha", "batch_size", "balance"]).ratio.mean()
          .unstack().reset_index())
    sections = [
        ("Q1: latency per alpha, 2 GPUs, mean of 3 seeds", markdown(q1.reset_index(), ".3f")),
        ("Q1: busier GPU's share of assignments at the largest batch",
         markdown(hot_share.pivot(index="alpha", columns="seed", values="share").reset_index(), ".3f")),
        ("Q1: compute time / step latency, mean of 3 seeds", markdown(shares, ".2f")),
        ("Q1: median phase times at the largest batch, seed 471 (ms)", markdown(phases.reset_index(), ".3f")),
        ("Q2: latency relative to No Load Balance, mean of 3 seeds", markdown(q2, ".2f")),
        ("Q2: median phase times at the largest 2-GPU batch, alpha 1.5, seed 471 (ms)",
         markdown(q2_phases.reset_index(), ".3f")),
        ("Q3: R^2 of layer latency on each load measure, per batch, across configurations", markdown(r2, ".3f")),
        ("Timeline: GPU busy during compute and derived GEMM rates", markdown(timeline, ".1f")),
        ("Contention: diagnostic runs, seed 471, No Load Balance, alpha 1.5 (slowest GPU per step, ms)",
         markdown(diag_runs, ".3f")),
        ("Contention: % of steps with dispatch over 3x its configuration's median + 0.2 ms",
         markdown(stall_shares, ".1f")),
        ("Contention: median latency of the same configuration in the 4-GPU sweep and the diagnostic runs (ms)",
         markdown(inflation, ".3f")),
    ]
    text = "\n\n".join(f"## {title}\n\n{body}" for title, body in sections)
    (out / "tables.md").write_text(text + "\n")
    print(text)
    print(f"\nWrote figures and tables.md to {out}/")


if __name__ == "__main__":
    main()
