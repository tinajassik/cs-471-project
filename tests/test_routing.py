import pytest
import torch

from moe_bench.routing import (
    contiguous_placement,
    load_metrics,
    sample_assignments,
    zipf_probabilities,
)


def test_alpha_zero_is_uniform():
    probabilities = zipf_probabilities(60, 0.0, seed=1)
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert torch.allclose(probabilities, torch.full((60,), 1 / 60, dtype=torch.float64))


def test_zipf_follows_rank_power_law():
    probabilities = zipf_probabilities(60, 1.5, seed=1).sort(descending=True).values
    assert probabilities[0] / probabilities[1] == pytest.approx(2**1.5)
    assert probabilities[0] / probabilities[3] == pytest.approx(4**1.5)


def test_hot_experts_are_shuffled_across_gpus():
    # Otherwise contiguous placement would put every hot expert on GPU 0.
    owner = contiguous_placement(60, 2)
    hottest = zipf_probabilities(60, 1.5, seed=471).topk(10).indices
    assert owner[hottest].unique().numel() == 2


def test_assignments_are_distinct_per_token():
    generator = torch.Generator().manual_seed(0)
    probabilities = zipf_probabilities(60, 3.0, seed=1)
    assignments = sample_assignments(probabilities, 500, 4, generator)
    assert assignments.shape == (500, 4)
    assert all(row.unique().numel() == 4 for row in assignments)


def test_contiguous_placement_matches_proposal():
    owner = contiguous_placement(60, 2)
    assert owner[:30].eq(0).all() and owner[30:].eq(1).all()
    assert torch.bincount(contiguous_placement(60, 4)).tolist() == [15, 15, 15, 15]


def test_load_metrics_on_a_known_batch():
    # GPU 0 owns experts 0-1, and GPU 1 owns experts 2-3.
    owner = contiguous_placement(4, 2)
    assignments = torch.tensor([[0, 1], [0, 2], [0, 1]])
    metrics = load_metrics(assignments, owner, 2)
    assert metrics.expert_imbalance == pytest.approx(3 / 1.5)
    assert metrics.gpu_imbalance == pytest.approx(5 / 3)
    assert metrics.tokens_per_gpu == [5, 1]
    assert metrics.activated_per_gpu == [2, 1]
    assert metrics.max_tokens_per_gpu == 5
    assert metrics.max_activated_per_gpu == 2
