import csv

import numpy as np
import pytest

from analysis.analyse_results import cost_fit, load, per_step
from moe_bench.benchmark import RAW_FIELDS


def write_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_rows():
    # Compute follows a known rule, so the fit must recover it exactly.
    generator = np.random.default_rng(0)
    rows = []
    for step in range(50):
        for rank in (0, 1):
            activated = int(generator.integers(1, 31))
            tokens = activated * int(generator.integers(1, 40))
            compute = 0.060 + 0.055 * activated + 0.0002 * tokens
            rows.append({
                "seed": 471, "expert_size": 1408, "warm_passes": 1, "alpha": 1.0, "batch_size": 64,
                "layers": 1, "world_size": 2, "step": step, "rank": rank,
                "dispatch_ms": 0.1, "compute_ms": compute, "combine_ms": 0.1, "total_ms": compute + 0.2,
                "tokens": tokens, "activated": activated, "bytes_sent": 0, "new_sizes": 0,
                "expert_imbalance": 1.0, "gpu_imbalance": 1.0,
                "max_tokens_per_gpu": tokens, "max_activated_per_gpu": activated,
            })
    return rows


def test_cost_fit_recovers_known_costs(tmp_path):
    path = tmp_path / "run.csv"
    write_csv(path, make_rows())
    data = load(path)
    constant, per_expert, per_token, r2 = cost_fit(data, data["batch_size"] <= 64)
    assert constant == pytest.approx(60, abs=0.01)
    assert per_expert == pytest.approx(55, abs=0.01)
    assert per_token == pytest.approx(0.2, abs=0.001)
    assert r2 == pytest.approx(1.0)


def test_cost_fit_does_not_invent_a_per_expert_cost_when_activated_is_constant(tmp_path):
    path = tmp_path / "run.csv"
    rows = make_rows()
    for row in rows:
        row["activated"] = 30
        row["compute_ms"] = 1.7 + 0.0001 * row["tokens"]
    write_csv(path, rows)
    data = load(path)
    constant, per_expert, per_token, r2 = cost_fit(data, data["batch_size"] <= 64)
    assert np.isnan(per_expert)
    assert constant == pytest.approx(1700, abs=0.01)
    assert per_token == pytest.approx(0.1, abs=0.001)
    assert r2 == pytest.approx(1.0)


def test_step_latency_is_the_slowest_gpu(tmp_path):
    path = tmp_path / "run.csv"
    rows = make_rows()
    write_csv(path, rows)
    data = load(path)
    latency, compute, _, _ = per_step(data, data["batch_size"] == 64)
    expected = max(rows[0]["total_ms"], rows[1]["total_ms"])
    assert latency[0] == pytest.approx(expected)
    assert compute[0] == pytest.approx(expected - 0.2)
