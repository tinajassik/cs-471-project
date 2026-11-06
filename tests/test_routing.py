import pytest
import torch

from moe_bench.routing import (
    contiguous_placement,
    eplb_placement,
    load_metrics,
    measured_load,
    route_to_replicas,
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


def test_load_metrics_counts_copies_but_keeps_expert_imbalance_logical():
    # Physical experts 0 and 2 are copies of logical expert 0, one on each GPU.
    physical_to_logical = torch.tensor([0, 1, 0, 2])
    owner = contiguous_placement(4, 2)
    physical = torch.tensor([[0, 1], [2, 3], [0, 1]])
    metrics = load_metrics(physical, owner, 2, physical_to_logical)
    assert metrics.tokens_per_gpu == [4, 2]
    assert metrics.activated_per_gpu == [2, 2]
    assert metrics.expert_imbalance == pytest.approx(3 / 2)  # expert 0 has 3 of the 6 assignments


def test_measured_load_shows_the_cap_on_hot_experts():
    probabilities = zipf_probabilities(60, 1.5, seed=1)
    load = measured_load(probabilities, 4, seed=2)
    assert load.sum() == 100_000 * 4
    share = load / load.sum()
    assert share.argmax() == probabilities.argmax()
    # A token picks an expert at most once, so no expert can exceed 1/top_k.
    assert probabilities.max() > 0.4
    assert share.max() <= 0.25


def test_eplb_worked_example():
    # Worked by hand. Copies: expert 0 goes from 8 to 4 to 2.67 per copy, ending with 3.
    # Packing 4, 2.67, 2.67, 2.67, 2, 2 into 3 slots per GPU gives loads 8.67 and 7.33.
    load = torch.tensor([8.0, 4.0, 2.0, 2.0])
    physical_to_logical, owner = eplb_placement(load, 6, 2)
    assert torch.bincount(physical_to_logical).tolist() == [3, 1, 1, 1]
    assert owner.tolist() == [0, 0, 0, 1, 1, 1]
    per_copy = (load / torch.bincount(physical_to_logical))[physical_to_logical]
    gpu_load = [per_copy[owner == gpu].sum().item() for gpu in (0, 1)]
    assert sorted(gpu_load) == pytest.approx([22 / 3, 26 / 3])


def test_eplb_without_extra_slots_only_moves_experts():
    load = measured_load(zipf_probabilities(60, 1.5, seed=1), 4, seed=2)
    physical_to_logical, owner = eplb_placement(load, 60, 2)
    assert physical_to_logical.sort().values.tolist() == list(range(60))
    assert torch.bincount(owner).tolist() == [30, 30]


@pytest.mark.parametrize("num_physical, world_size", [(60, 2), (90, 2), (60, 4), (92, 4)])
def test_eplb_balances_tokens_better_than_contiguous(num_physical, world_size):
    # Seed 471 puts about 79% of tokens on one GPU under contiguous placement on 2 GPUs.
    probabilities = zipf_probabilities(60, 1.5, seed=471)
    load = measured_load(probabilities, 4, seed=1)
    physical_to_logical, owner = eplb_placement(load, num_physical, world_size)
    generator = torch.Generator().manual_seed(3)
    assignments = sample_assignments(probabilities, 16384, 4, generator)
    physical = route_to_replicas(assignments, physical_to_logical, generator)

    before = load_metrics(assignments, contiguous_placement(60, world_size), world_size).gpu_imbalance
    after = load_metrics(physical, owner, world_size, physical_to_logical).gpu_imbalance
    assert before > 1.4
    assert after < 1.05


def test_eplb_gives_the_hottest_expert_the_most_copies():
    load = measured_load(zipf_probabilities(60, 1.5, seed=1), 4, seed=2)
    physical_to_logical, _ = eplb_placement(load, 90, 2)
    copies = torch.bincount(physical_to_logical, minlength=60)
    assert copies.sum() == 90
    assert copies.min() == 1
    assert copies.argmax() == load.argmax()


def test_eplb_rejects_slots_that_do_not_divide_across_gpus():
    with pytest.raises(ValueError):
        eplb_placement(torch.ones(60), 91, 2)


def test_eplb_matches_the_open_source_code():
    # Skipped unless DeepSeek's eplb.py is importable: see the README.
    eplb = pytest.importorskip("eplb")
    for alpha in (0.5, 1.0, 1.5):
        for num_physical, num_gpus in ((60, 2), (90, 2), (60, 4), (92, 4)):
            load = measured_load(zipf_probabilities(60, alpha, seed=1), 4, seed=2)
            ours, _ = eplb_placement(load, num_physical, num_gpus)
            theirs = eplb.rebalance_experts(load.unsqueeze(0), num_physical, 1, 1, num_gpus)[0][0]
            assert ours.tolist() == theirs.tolist()


def test_replicas_split_each_experts_tokens_evenly():
    physical_to_logical = torch.tensor([0, 1, 0, 2, 0, 3])
    generator = torch.Generator().manual_seed(0)
    assignments = torch.tensor([[0, 1]] * 100 + [[0, 2]] * 50)
    physical = route_to_replicas(assignments, physical_to_logical, generator)

    assert physical.shape == assignments.shape
    assert physical_to_logical[physical].tolist() == assignments.tolist()
    counts = torch.bincount(physical.flatten(), minlength=6)
    assert counts[[0, 2, 4]].tolist() == [50, 50, 50]
    assert counts[1] == 100 and counts[3] == 50


def test_replica_split_is_the_same_on_every_rank():
    physical_to_logical = torch.tensor([0, 1, 0, 2, 0, 3])
    assignments = torch.tensor([[0, 1], [0, 2], [3, 0]])
    first = route_to_replicas(assignments, physical_to_logical, torch.Generator().manual_seed(5))
    second = route_to_replicas(assignments, physical_to_logical, torch.Generator().manual_seed(5))
    assert first.tolist() == second.tolist()


def test_single_token_steps_do_not_always_use_the_same_copy():
    # Otherwise small batches would all go to the GPU holding the first copy.
    physical_to_logical = torch.tensor([0, 0])
    generator = torch.Generator().manual_seed(0)
    used = {route_to_replicas(torch.tensor([[0]]), physical_to_logical, generator).item() for _ in range(50)}
    assert used == {0, 1}


def test_without_replicas_routing_is_unchanged():
    assignments = torch.tensor([[3, 1], [0, 2]])
    physical = route_to_replicas(assignments, torch.arange(4), torch.Generator().manual_seed(0))
    assert physical.tolist() == assignments.tolist()
