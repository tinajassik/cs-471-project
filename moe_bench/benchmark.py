"""MoE-layer benchmark with Zipf or recorded routing: No Load Balance, or EPLB placement with replicas.

Launch one process per GPU:
    torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark --alphas 0,1 --batch-sizes 1,64,1024
Replay a real model's routing, recorded by model_trace.py, at that model's expert shape:
    torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark --trace results/trace_qwen3.npz \
        --trace-layers 0,23 --trace-phase prefill --batch-sizes 1024,65536
"""
import argparse
import csv
import dataclasses
import math
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from .moe_layer import (
    ExpertShape,
    MoELayer,
    build_plan,
    outputs_match,
    reference_output,
    token_homes,
)
from .routing import (
    ZipfRouting,
    contiguous_placement,
    eplb_placement,
    load_metrics,
    load_trace,
    route_to_replicas,
)

PHASES = ("dispatch", "compute", "combine")
NEXT_PHASE = {"start": "dispatch", "dispatch": "compute", "compute": "combine"}
# Physical experts per logical expert under each balancing policy.
EPLB_REPLICATION = {"eplb-1.0": 1.0, "eplb-1.5": 1.5}
# Expert shapes of the models we trace. A trace carries its own shape, so --trace ignores these.
SHAPES = {
    "qwen1.5-moe": ExpertShape(),
    "qwen3-30b-a3b": ExpertShape(num_experts=128, top_k=8, hidden_size=2048, expert_size=768),
}
RAW_FIELDS = [
    "seed", "expert_size", "num_experts", "top_k", "warm_passes", "balance", "physical_experts",
    "routing", "alpha", "batch_size", "layers", "world_size", "step", "rank",
    "dispatch_ms", "compute_ms", "combine_ms", "total_ms",
    "tokens", "activated", "bytes_sent", "new_sizes",
    "expert_imbalance", "gpu_imbalance", "max_tokens_per_gpu", "max_activated_per_gpu",
]


class EventPhases:
    """Times each phase with CUDA events, and labels it for Nsight Systems with NVTX."""

    def __init__(self):
        self.events = {}

    def mark(self, name):
        if name != "start":
            torch.cuda.nvtx.range_pop()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[name] = event
        if name in NEXT_PHASE:
            torch.cuda.nvtx.range_push(NEXT_PHASE[name])

    def durations_ms(self):
        previous = self.events["start"]
        durations = {}
        for name in PHASES:
            durations[name] = previous.elapsed_time(self.events[name])
            previous = self.events[name]
        durations["total"] = sum(durations.values())
        return durations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alphas", default="0,0.5,1,1.5", help="Zipf skew values, comma-separated")
    parser.add_argument("--batch-sizes", default="1,4,16,64,256,1024,4096",
                        help="total tokens per step across all GPUs, comma-separated")
    parser.add_argument("--steps", type=int, default=100, help="measured steps per configuration")
    parser.add_argument("--warmup", type=int, default=20, help="unmeasured steps per configuration")
    parser.add_argument("--warm-passes", type=int, default=1,
                        help="unmeasured replays of the measured routing, so timed steps meet no "
                             "new expert token counts; 0 keeps the first-time cost in the timings")
    parser.add_argument("--shape", default="qwen1.5-moe",
                        help=f"expert count, top-k and expert sizes for Zipf routing: {', '.join(SHAPES)}, or a "
                             "trace .npz to use its model's shape (for a uniform control next to a replay)")
    parser.add_argument("--top-k", type=int, default=None, help="override the shape's top-k (Zipf routing only)")
    parser.add_argument("--expert-size", type=int, default=None,
                        help="expert inner width; smaller means fewer weight bytes but the same operations")
    parser.add_argument("--trace", type=Path, default=None,
                        help="replay routing recorded by model_trace.py instead of Zipf; --alphas is then unused")
    parser.add_argument("--trace-layers", default="0", help="MoE layers of the trace to replay, comma-separated")
    parser.add_argument("--trace-phase", default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--balance", default="none", choices=["none", *EPLB_REPLICATION],
                        help="none is contiguous placement; eplb-1.0 only moves experts; eplb-1.5 also adds "
                             "replicas of hot experts")
    parser.add_argument("--layers", type=int, default=1,
                        help="separate MoE layers to cycle through, so weights are not all in cache")
    parser.add_argument("--seed", type=int, default=471)
    parser.add_argument("--output", type=Path, default=Path("results/baseline.csv"))
    parser.add_argument("--verify", action="store_true",
                        help="check outputs against a single-GPU reference (slower; don't use for timing)")
    args = parser.parse_args()
    args.alphas = [float(a) for a in args.alphas.split(",")]
    args.batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    args.trace_layers = [int(layer) for layer in args.trace_layers.split(",")]
    if args.trace and args.top_k:
        parser.error("--top-k can't change a trace: its tokens chose a fixed number of experts")
    if args.shape not in SHAPES and not args.shape.endswith(".npz"):
        parser.error(f"--shape must be one of {', '.join(SHAPES)} or a trace .npz")
    return args


def make_shape(args, trace_meta=None):
    """The model's expert shape: from the trace if there is one, else from --shape and overrides."""
    if trace_meta is None and args.shape.endswith(".npz"):
        trace_meta = trace_shape_meta(args.shape)
    if trace_meta:
        shape = ExpertShape(num_experts=trace_meta["num_experts"], top_k=trace_meta["top_k"],
                            hidden_size=trace_meta["hidden_size"], expert_size=trace_meta["moe_intermediate_size"])
    else:
        shape = SHAPES[args.shape]
    if args.top_k:
        shape = dataclasses.replace(shape, top_k=args.top_k)
    if args.expert_size:
        shape = dataclasses.replace(shape, expert_size=args.expert_size)
    return shape


def trace_shape_meta(path):
    import json

    import numpy as np

    with np.load(path) as trace:
        return json.loads(str(trace["meta"]))


def make_routings(args, shape):
    """One routing per configuration: a Zipf skew per alpha, or a trace layer per layer."""
    if args.trace:
        return [load_trace(args.trace, layer, args.trace_phase)[0] for layer in args.trace_layers]
    return [ZipfRouting(shape.num_experts, shape.top_k, alpha, args.seed) for alpha in args.alphas]


def make_placement(args, shape, world_size, routing):
    """Return (physical_to_logical, owner). EPLB's depends on the load, so on the routing."""
    if args.balance == "none":
        return torch.arange(shape.num_experts), contiguous_placement(shape.num_experts, world_size)
    load = routing.load()
    # Round up, because every GPU holds the same number: 1.5x on 4 GPUs is 92, not 90.
    per_gpu = math.ceil(shape.num_experts * EPLB_REPLICATION[args.balance] / world_size)
    return eplb_placement(load, per_gpu * world_size, world_size)


def make_steps(args, shape, physical_to_logical, owner, world_size, rank, routing, batch_size, device):
    """Build the same routing trace on every rank, from the shared seed.

    Replica choice has its own generator, so all policies see the same logical routing.
    Returns (logical assignments, physical assignments, plan) for each step.
    """
    replica_generator = torch.Generator().manual_seed(args.seed + 1001)
    steps = []
    for assignments in routing.batches(batch_size, args.warmup + args.steps, args.seed + 1):
        physical = route_to_replicas(assignments, physical_to_logical, replica_generator)
        steps.append((assignments, physical, build_plan(physical, owner, world_size, rank).to(device)))
    return steps


def run_config(args, shape, layers, physical_to_logical, owner, world_size, rank, routing, batch_size, seen_sizes,
               device="cuda"):
    """Run one (routing, batch size) point and return this rank's per-step rows.

    seen_sizes holds the expert token counts this process has already computed.
    Each new count cost about 1 ms in the first sweep, so warm passes replay the
    measured routing first, and new_sizes records any left in a timed step.
    """
    steps = make_steps(args, shape, physical_to_logical, owner, world_size, rank, routing, batch_size, device)
    warmup_steps, measured_steps = steps[:args.warmup], steps[args.warmup:]
    is_home = token_homes(batch_size, world_size) == rank
    input_generator = torch.Generator().manual_seed(args.seed + 2 + rank)
    x = torch.randn(int(is_home.sum()), shape.hidden_size, generator=input_generator)
    x = x.to(dtype=torch.bfloat16, device=device)

    def run_step(index, plan):
        sizes = {positions.numel() for _, positions in plan.expert_groups}
        new_sizes = len(sizes - seen_sizes)
        seen_sizes.update(sizes)
        dist.barrier()
        torch.cuda.synchronize()

        phases = EventPhases()
        output = layers[index % len(layers)](x, plan, phases=phases)
        torch.cuda.synchronize()
        return phases, output, new_sizes

    for index, (_, _, plan) in enumerate(warmup_steps):
        run_step(index, plan)
    for _ in range(args.warm_passes):
        for index, (_, _, plan) in enumerate(measured_steps):
            run_step(index, plan)

    rows = []
    for step, (assignments, physical, plan) in enumerate(measured_steps):
        phases, output, new_sizes = run_step(step, plan)

        if args.verify:
            home_assignments = assignments[is_home]
            layer = step % len(layers)
            expected = reference_output(x, home_assignments, shape, layer, args.seed)
            assert outputs_match(output, expected), (
                f"rank {rank}: wrong output at {routing.label} alpha={routing.alpha} batch={batch_size} step={step}"
            )

        durations = phases.durations_ms()
        metrics = load_metrics(physical, owner, world_size, physical_to_logical)
        rows.append({
            "seed": args.seed,
            "expert_size": shape.expert_size,
            "num_experts": shape.num_experts,
            "top_k": shape.top_k,
            "warm_passes": args.warm_passes,
            "balance": args.balance,
            "physical_experts": owner.numel(),
            "routing": routing.label,
            "alpha": routing.alpha,
            "batch_size": batch_size,
            "layers": len(layers),
            "world_size": world_size,
            "step": step,
            "rank": rank,
            **{f"{name}_ms": value for name, value in durations.items()},
            "tokens": metrics.tokens_per_gpu[rank],
            "activated": metrics.activated_per_gpu[rank],
            "bytes_sent": plan.send_token_idx.numel() * shape.hidden_size * 2,
            "new_sizes": new_sizes,
            "expert_imbalance": metrics.expert_imbalance,
            "gpu_imbalance": metrics.gpu_imbalance,
            "max_tokens_per_gpu": metrics.max_tokens_per_gpu,
            "max_activated_per_gpu": metrics.max_activated_per_gpu,
        })
    return rows


def print_summary(rows, world_size):
    """Print the median step latency, where each step takes as long as its slowest GPU."""
    by_step = {}
    for row in rows:
        by_step.setdefault(row["step"], []).append(row)
    step_latency = [max(r["total_ms"] for r in group) for group in by_step.values()]
    latency = statistics.median(step_latency)
    per_rank = "  ".join(
        f"gpu{r}: compute {statistics.median(x['compute_ms'] for x in rows if x['rank'] == r):.3f} ms"
        for r in range(world_size)
    )
    first = rows[0]
    throughput = first["batch_size"] / (latency / 1000)
    print(
        f"{first.get('routing', 'zipf')} alpha={first['alpha']:<4} batch={first['batch_size']:<5} "
        f"latency={latency:.3f} ms  throughput={throughput:,.0f} tokens/s  "
        f"gpu_imbalance={statistics.median(r['gpu_imbalance'] for r in rows):.2f}  "
        f"new_sizes={sum(r['new_sizes'] for r in rows)}  {per_rank}",
        flush=True,
    )


def main():
    args = parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    trace_meta = load_trace(args.trace, args.trace_layers[0], args.trace_phase)[1] if args.trace else None
    shape = make_shape(args, trace_meta)
    routings = make_routings(args, shape)
    writer = None
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output.open("w", newline="")
        writer = csv.DictWriter(output_file, RAW_FIELDS)
        writer.writeheader()
        expert_mb = 3 * shape.hidden_size * shape.expert_size * 2 / 1e6
        print(f"{world_size} GPUs ({torch.cuda.get_device_name()}), {shape}, {expert_mb:.1f} MB per expert, "
              f"{args.layers} layer(s), seed={args.seed}, balance={args.balance}, "
              f"warm_passes={args.warm_passes}, verify={args.verify}", flush=True)
        if args.trace:
            print(f"replaying {args.trace_phase} routing of {trace_meta['model']} from {args.trace}: "
                  + ", ".join(f"layer {r.layer} ({r.assignments.shape[0]} tokens)" for r in routings), flush=True)

    layers = None
    seen_sizes = set()
    for routing in routings:
        # The baseline's placement is the same for every routing, so it keeps its weights.
        if layers is None or args.balance != "none":
            layers = None
            torch.cuda.empty_cache()
            physical_to_logical, owner = make_placement(args, shape, world_size, routing)
            layers = [
                MoELayer(shape, owner, rank, layer_index, args.seed, physical_to_logical).to("cuda")
                for layer_index in range(args.layers)
            ]
            if rank == 0 and args.balance != "none":
                copies = torch.bincount(physical_to_logical).sort(descending=True).values[:6].tolist()
                print(f"{routing.label} alpha={routing.alpha}: {owner.numel()} physical experts, copies of the 6 most copied: {copies}",
                      flush=True)
        for batch_size in args.batch_sizes:
            with torch.inference_mode():
                rows = run_config(args, shape, layers, physical_to_logical, owner, world_size, rank, routing,
                                  batch_size, seen_sizes)
            gathered = [None] * world_size
            dist.all_gather_object(gathered, rows)
            if rank == 0:
                all_rows = [row for rank_rows in gathered for row in rank_rows]
                writer.writerows(all_rows)
                output_file.flush()
                print_summary(all_rows, world_size)

    if rank == 0:
        output_file.close()
        print(f"BENCHMARK FINISHED, wrote {args.output}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
