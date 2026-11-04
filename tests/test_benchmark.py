"""Runs one benchmark configuration on the CPU, with the GPU-only calls replaced."""
import argparse
import csv
import io

import pytest
import torch

from moe_bench import benchmark
from moe_bench.moe_layer import ExpertShape, MoELayer
from moe_bench.routing import contiguous_placement

SHAPE = ExpertShape(num_experts=8, top_k=2, hidden_size=32, expert_size=16)


class FakePhases:
    def mark(self, name):
        pass

    def durations_ms(self):
        return {"dispatch": 1.0, "compute": 2.0, "combine": 1.0, "total": 4.0}


@pytest.fixture
def cpu_run(monkeypatch):
    monkeypatch.setattr(benchmark, "EventPhases", FakePhases)
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark.dist, "barrier", lambda: None)
    args = argparse.Namespace(seed=1, warmup=2, steps=5, verify=True)
    owner = contiguous_placement(SHAPE.num_experts, 1)
    layers = [MoELayer(SHAPE, owner, 0, i, args.seed) for i in range(2)]
    with torch.inference_mode():
        return benchmark.run_config(args, SHAPE, layers, owner, 1, 0, 1.0, 16, device="cpu")


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
                "batch_size": 100, "layers": 1, "gpu_imbalance": 1.5}

    benchmark.print_summary([row(0, 2.0), row(1, 10.0)], 2)
    out = capsys.readouterr().out
    assert "latency=10.000 ms" in out
    assert "throughput=10,000 tokens/s" in out
