"""Replaying recorded routing in the benchmark, and the trace and replay analyses."""
import argparse
import csv
import json

import numpy as np
import pytest
import torch

from analysis import analyse_replay, analyse_trace
from moe_bench import benchmark
from moe_bench.moe_layer import ExpertShape
from moe_bench.routing import TraceRouting, ZipfRouting, load_trace
from tests.test_benchmark import SHAPE, run_cpu

QWEN3 = {"model": "Qwen/Qwen3-30B-A3B", "num_experts": 128, "top_k": 8, "hidden_size": 2048,
         "moe_intermediate_size": 768}


def recorded(num_tokens=200, alpha=1.0, seed=3, shape=SHAPE):
    """Stand-in for a recorded layer: Zipf choices, so the true skew is known."""
    return ZipfRouting(shape.num_experts, shape.top_k, alpha, seed).batches(num_tokens, 1, seed)[0]


def write_trace(path, prefill, decode, decode_prompt, decode_step, meta=None, sources=("gsm8k", "humaneval")):
    meta = meta or {"model": "tiny", "num_experts": SHAPE.num_experts, "top_k": SHAPE.top_k,
                    "hidden_size": SHAPE.hidden_size, "moe_intermediate_size": SHAPE.expert_size}
    np.savez(path, layers=np.array([0]), meta=np.array(json.dumps(meta)), prompt_source=np.array(sources),
             prefill_0=prefill.numpy().astype(np.uint8), decode_0=decode.numpy().astype(np.uint8),
             prefill_prompt=np.repeat(np.arange(len(sources)), len(prefill) // len(sources)).astype(np.int32),
             decode_prompt=np.asarray(decode_prompt, np.int32), decode_step=np.asarray(decode_step, np.int32))


def test_trace_batches_are_consecutive_recorded_tokens():
    tokens = recorded(10)
    routing = TraceRouting(tokens, SHAPE.num_experts, layer=3, phase="prefill")
    batches = routing.batches(4, 20, seed=1)
    assert len(batches) == 20 and all(b.shape == (4, SHAPE.top_k) for b in batches)
    rows = {tuple(row) for row in tokens.tolist()}
    for batch in batches:
        start = next(i for i in range(10) if tokens[i].tolist() == batch[0].tolist())
        assert batch.tolist() == tokens[(start + torch.arange(4)) % 10].tolist()
        assert {tuple(row) for row in batch.tolist()} <= rows
    assert all(torch.equal(a, b) for a, b in zip(batches, routing.batches(4, 20, seed=1)))


def test_trace_batches_wrap_when_larger_than_the_trace():
    routing = TraceRouting(recorded(5), SHAPE.num_experts, 0, "prefill")
    assert routing.batches(12, 1, seed=0)[0].shape == (12, SHAPE.top_k)


def test_trace_load_is_the_recorded_count():
    tokens = recorded(50)
    routing = TraceRouting(tokens, SHAPE.num_experts, 0, "prefill")
    assert routing.load().tolist() == torch.bincount(tokens.flatten(), minlength=SHAPE.num_experts).double().tolist()
    assert routing.label == "trace-prefill-L0" and np.isnan(routing.alpha)


def test_decode_is_replayed_step_by_step(tmp_path):
    # Two prompts, recorded one after the other: prompt 0 steps 1-3, then prompt 1 steps 1-2.
    decode = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [0, 7]])
    write_trace(tmp_path / "t.npz", recorded(4), decode, [0, 0, 0, 1, 1], [1, 2, 3, 1, 2])
    routing, _ = load_trace(tmp_path / "t.npz", 0, "decode")
    assert routing.assignments.tolist() == [[0, 1], [6, 7], [2, 3], [0, 7], [4, 5]]


def test_missing_layer_is_an_error(tmp_path):
    write_trace(tmp_path / "t.npz", recorded(4), recorded(2), [0, 1], [1, 1])
    with pytest.raises(ValueError, match="no MoE layer 5"):
        load_trace(tmp_path / "t.npz", 5, "prefill")


@pytest.mark.parametrize("balance", ["none", "eplb-1.0", "eplb-1.5"])
def test_trace_replay_passes_the_output_check(monkeypatch, balance):
    routing = TraceRouting(recorded(300), SHAPE.num_experts, 7, "prefill")
    rows = run_cpu(monkeypatch, warm_passes=1, balance=balance, routing=routing)
    assert {row["routing"] for row in rows} == {"trace-prefill-L7"}
    assert all(np.isnan(row["alpha"]) for row in rows)
    assert rows[0]["top_k"] == SHAPE.top_k and rows[0]["num_experts"] == SHAPE.num_experts


def test_eplb_balances_on_the_recorded_load():
    routing = TraceRouting(recorded(2000, alpha=1.5), SHAPE.num_experts, 0, "prefill")
    args = argparse.Namespace(balance="eplb-1.0", seed=1)
    physical_to_logical, owner = benchmark.make_placement(args, SHAPE, 2, routing)
    gpu_load = torch.bincount(owner, weights=routing.load()[physical_to_logical], minlength=2)
    contiguous = torch.bincount(torch.arange(8) * 2 // 8, weights=routing.load(), minlength=2)
    assert gpu_load.max() / gpu_load.mean() <= contiguous.max() / contiguous.mean()


def shape_args(**overrides):
    values = dict(shape="qwen1.5-moe", top_k=None, expert_size=None)
    return argparse.Namespace(**{**values, **overrides})


def test_shape_comes_from_the_trace():
    assert benchmark.make_shape(shape_args(), QWEN3) == ExpertShape(128, 8, 2048, 768)
    assert benchmark.make_shape(shape_args(), QWEN3) == benchmark.SHAPES["qwen3-30b-a3b"]


def test_shape_can_come_from_a_trace_file(tmp_path):
    write_trace(tmp_path / "t.npz", recorded(4), recorded(2), [0, 1], [1, 1], meta=QWEN3)
    assert benchmark.make_shape(shape_args(shape=str(tmp_path / "t.npz"))) == benchmark.SHAPES["qwen3-30b-a3b"]


def test_shape_presets_and_overrides():
    assert benchmark.make_shape(shape_args()) == ExpertShape()
    assert benchmark.make_shape(shape_args(top_k=8)).top_k == 8
    assert benchmark.make_shape(shape_args(shape="qwen3-30b-a3b", expert_size=12)).expert_size == 12


def test_physical_experts_for_the_larger_model():
    routing = ZipfRouting(128, 8, 1.5, 1)
    for balance, world_size, expected in (("eplb-1.0", 4, 128), ("eplb-1.5", 2, 192), ("eplb-1.5", 4, 192)):
        _, owner = benchmark.make_placement(argparse.Namespace(balance=balance, seed=1), ExpertShape(128, 8),
                                            world_size, routing)
        assert owner.numel() == expected


@pytest.mark.parametrize("alpha", [0.0, 0.8, 1.5])
def test_alpha_fit_recovers_the_skew(alpha):
    counts = analyse_trace.expert_counts(recorded(40_000, alpha=alpha, shape=ExpertShape(60, 4)).numpy(), 60)
    fitted, distance = analyse_trace.fit_alpha(counts, 4)
    assert abs(fitted - alpha) <= 0.06
    assert distance < analyse_trace.POOR_FIT


def test_noise_alpha_shrinks_with_more_tokens():
    assert analyse_trace.noise_alpha(100, 60, 4) > analyse_trace.noise_alpha(50_000, 60, 4)
    assert analyse_trace.noise_alpha(50_000, 60, 4) <= 0.04


def test_alpha_fit_flags_a_shape_zipf_cannot_make():
    # Two very hot experts and the rest equal: no Zipf curve looks like this.
    counts = np.array([1000.0, 1000.0] + [10.0] * 58)
    assert analyse_trace.fit_alpha(counts, 4)[1] > analyse_trace.POOR_FIT


def test_load_measures():
    counts = np.array([3.0, 1.0, 0.0, 0.0])
    assert analyse_trace.gpu_imbalance(counts, 2) == pytest.approx(2.0)
    assert analyse_trace.gini(np.ones(5)) == pytest.approx(0.0)
    assert analyse_trace.normalised_entropy(np.ones(8)) == pytest.approx(1.0)
    assert analyse_trace.spearman(np.array([1, 2, 3.0]), np.array([10, 20, 30.0])) == pytest.approx(1.0)
    assert analyse_trace.spearman(np.array([1, 2, 3.0]), np.array([3, 2, 1.0])) == pytest.approx(-1.0)
    assert analyse_trace.top_overlap(np.arange(10.0), np.arange(10.0), n=5) == 1.0


def test_trace_analysis_runs_end_to_end(tmp_path, monkeypatch, capsys):
    path = tmp_path / "trace_tiny.npz"
    write_trace(path, recorded(400), recorded(6), [0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3])
    with open(tmp_path / "trace_tiny_timing.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, ["prompt", "source", "phase", "step", "tokens", "context",
                                       "total_ms", "attention_ms", "moe_ms", "other_ms"])
        writer.writeheader()
        writer.writerow({"prompt": 0, "source": "gsm8k", "phase": "prefill", "step": 0, "tokens": 200, "context": 200,
                         "total_ms": 10, "attention_ms": 3, "moe_ms": 6, "other_ms": 1})
        writer.writerow({"prompt": 0, "source": "gsm8k", "phase": "decode", "step": 1, "tokens": 1, "context": 201,
                         "total_ms": 5, "attention_ms": 1, "moe_ms": 3, "other_ms": 1})
    monkeypatch.setattr("sys.argv", ["analyse_trace.py", str(path)])
    analyse_trace.main()
    out = capsys.readouterr().out
    assert "MoE share" in out and "60.0%" in out
    assert (tmp_path / "trace_tiny_layers.csv").exists()
    monkeypatch.setattr("sys.argv", ["analyse_trace.py", str(path), "--pick-layers"])
    analyse_trace.main()
    assert capsys.readouterr().out.strip() == "0"


def test_replay_summary_compares_with_uniform_and_no_balance(tmp_path, monkeypatch, capsys):
    path = tmp_path / "replay.csv"
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, benchmark.RAW_FIELDS)
        writer.writeheader()
        for routing, alpha, balance, total in (("zipf", 0.0, "none", 2.0), ("trace-prefill-L3", float("nan"), "none", 3.0),
                                               ("trace-prefill-L3", float("nan"), "eplb-1.0", 2.4)):
            for step in range(3):
                for rank in (0, 1):
                    writer.writerow({"routing": routing, "alpha": alpha, "balance": balance, "world_size": 2,
                                     "top_k": 8, "batch_size": 1024, "step": step, "rank": rank,
                                     "total_ms": total if rank == 1 else 1.0, "gpu_imbalance": 1.2,
                                     "max_activated_per_gpu": 64})
    monkeypatch.setattr("sys.argv", ["analyse_replay.py", str(path)])
    analyse_replay.main()
    lines = [line.split() for line in capsys.readouterr().out.splitlines() if "trace-prefill-L3" in line]
    by_balance = {line[3]: line for line in lines}
    assert by_balance["none"][7:9] == ["1.50", "1.00"]
    assert by_balance["eplb-1.0"][7:9] == ["1.20", "0.80"]
