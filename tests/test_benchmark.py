"""Runs one benchmark configuration on the CPU, with the GPU-only calls replaced."""
import argparse
import csv
import io

import pytest
import torch

from moe_bench import benchmark
from moe_bench.moe_layer import ExpertShape, MoELayer
from moe_bench.routing import ZipfRouting, contiguous_placement

SHAPE = ExpertShape(num_experts=8, top_k=2, hidden_size=32, expert_size=16)


class FakePhases:
    def mark(self, name):
        pass

    def durations_ms(self):
        return {"dispatch": 1.0, "compute": 2.0, "combine": 1.0, "total": 4.0}


def make_args(warm_passes=1, warmup=2, balance="none"):
    return argparse.Namespace(seed=1, warmup=warmup, steps=5, verify=True, warm_passes=warm_passes, balance=balance)


def zipf(shape, alpha=1.0, seed=1):
    return ZipfRouting(shape.num_experts, shape.top_k, alpha, seed)


def run_cpu(monkeypatch, warm_passes, seen_sizes=None, shape=SHAPE, warmup=2, balance="none", routing=None,
            world_size=1):
    monkeypatch.setattr(benchmark, "EventPhases", FakePhases)
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark.dist, "barrier", lambda: None)
    args = make_args(warm_passes, warmup, balance)
    routing = routing or zipf(shape, seed=args.seed)
    physical_to_logical, owner = benchmark.make_placement(args, shape, world_size, routing)
    layers = [MoELayer(shape, owner, 0, i, args.seed, physical_to_logical) for i in range(2)]
    seen_sizes = set() if seen_sizes is None else seen_sizes
    with torch.inference_mode():
        return benchmark.run_config(args, shape, layers, physical_to_logical, owner, world_size, 0, routing, 16,
                                    seen_sizes, device="cpu")


@pytest.fixture
def cpu_run(monkeypatch):
    return run_cpu(monkeypatch, warm_passes=1)


def test_without_warm_up_the_first_step_reports_new_sizes(monkeypatch):
    rows = run_cpu(monkeypatch, warm_passes=0, warmup=0)
    assert rows[0]["new_sizes"] > 0


def test_warm_pass_removes_new_sizes_from_timed_steps(monkeypatch):
    rows = run_cpu(monkeypatch, warm_passes=1, warmup=0)
    assert [row["new_sizes"] for row in rows] == [0] * 5


def test_sizes_already_seen_are_not_counted_again(monkeypatch):
    seen = set()
    first = run_cpu(monkeypatch, warm_passes=0, warmup=0, seen_sizes=seen)
    assert sum(row["new_sizes"] for row in first) == len(seen)
    again = run_cpu(monkeypatch, warm_passes=0, warmup=0, seen_sizes=seen)
    assert [row["new_sizes"] for row in again] == [0] * 5


def test_smaller_expert_keeps_the_same_routing(monkeypatch):
    small = ExpertShape(num_experts=8, top_k=2, hidden_size=32, expert_size=4)
    rows_full = run_cpu(monkeypatch, warm_passes=1)
    rows_small = run_cpu(monkeypatch, warm_passes=1, shape=small)
    assert rows_small[0]["expert_size"] == 4
    for full, tiny in zip(rows_full, rows_small):
        assert (full["tokens"], full["activated"]) == (tiny["tokens"], tiny["activated"])


@pytest.mark.parametrize("balance", ["eplb-1.0", "eplb-1.5"])
def test_eplb_run_passes_the_output_check(monkeypatch, balance):
    # run_cpu has verify on, so a wrong output would raise here.
    rows = run_cpu(monkeypatch, warm_passes=1, balance=balance)
    assert {row["balance"] for row in rows} == {balance}
    assert rows[0]["physical_experts"] == {"eplb-1.0": 8, "eplb-1.5": 12}[balance]


def test_replicas_activate_more_expert_copies(monkeypatch):
    def mean_activated(balance):
        rows = run_cpu(monkeypatch, warm_passes=1, balance=balance)
        return sum(row["activated"] for row in rows) / len(rows)

    assert mean_activated("eplb-1.0") == mean_activated("none")
    assert mean_activated("eplb-1.5") > mean_activated("none")


def test_every_policy_sees_the_same_logical_routing():
    traces = {}
    for balance in ("none", "eplb-1.0", "eplb-1.5"):
        args = make_args(balance=balance)
        physical_to_logical, owner = benchmark.make_placement(args, SHAPE, 2, zipf(SHAPE))
        steps = benchmark.make_steps(args, SHAPE, physical_to_logical, owner, 2, 0, zipf(SHAPE), 16, "cpu")
        traces[balance] = [step[0] for step in steps]
        for assignments, physical, _ in steps:
            assert physical_to_logical[physical].tolist() == assignments.tolist()
    for balance in ("eplb-1.0", "eplb-1.5"):
        assert all(torch.equal(a, b) for a, b in zip(traces["none"], traces[balance]))


@pytest.mark.parametrize("balance, world_size, expected", [
    ("eplb-1.0", 2, 60), ("eplb-1.5", 2, 90), ("eplb-1.0", 4, 60), ("eplb-1.5", 4, 92),
])
def test_physical_expert_count_divides_across_gpus(balance, world_size, expected):
    physical_to_logical, owner = benchmark.make_placement(make_args(balance=balance), ExpertShape(), world_size,
                                                          zipf(ExpertShape(), 1.5))
    assert owner.numel() == physical_to_logical.numel() == expected
    assert torch.bincount(owner).tolist() == [expected // world_size] * world_size


def test_baseline_placement_is_contiguous_with_no_replicas():
    physical_to_logical, owner = benchmark.make_placement(make_args(), SHAPE, 2, zipf(SHAPE, 1.5))
    assert physical_to_logical.tolist() == list(range(8))
    assert owner.tolist() == contiguous_placement(8, 2).tolist()


def test_only_measured_steps_are_recorded(cpu_run):
    assert [row["step"] for row in cpu_run] == [0, 1, 2, 3, 4]


def test_rows_fit_the_csv_columns(cpu_run):
    writer = csv.DictWriter(io.StringIO(), benchmark.RAW_FIELDS)
    writer.writerows(cpu_run)
    assert set(cpu_run[0]) == set(benchmark.RAW_FIELDS)


def test_row_values(cpu_run):
    row = cpu_run[0]
    assert row["tokens"] == 16 * SHAPE.top_k
    assert row["bytes_sent"] == 0  # one GPU has nothing to send
    assert row["total_ms"] == 4.0
    assert row["layers"] == 2


def test_summary_uses_the_slowest_gpu(capsys):
    def row(rank, total):
        return {"step": 0, "rank": rank, "total_ms": total, "compute_ms": total, "alpha": 1.0,
                "batch_size": 100, "layers": 1, "gpu_imbalance": 1.5, "new_sizes": 0}

    benchmark.print_summary([row(0, 2.0), row(1, 10.0)], 2)
    out = capsys.readouterr().out
    assert "latency=10.000 ms" in out
    assert "throughput=10,000 tokens/s" in out
