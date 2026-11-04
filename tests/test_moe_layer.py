"""Runs the dispatch, compute and combine path with threads standing in for GPUs."""
import threading

import pytest
import torch

from moe_bench.moe_layer import (
    Expert,
    ExpertShape,
    MoELayer,
    build_plan,
    outputs_match,
    reference_output,
    token_homes,
)
from moe_bench.routing import contiguous_placement, sample_assignments, zipf_probabilities

# Smaller than Qwen1.5 so the tests are fast. Dispatch logic doesn't depend on size.
SHAPE = ExpertShape(num_experts=12, top_k=4, hidden_size=64, expert_size=32)
SEED = 7


class FakeNetwork:
    """All-to-all between threads, with the same argument meaning as the NCCL version."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.chunks = [None] * world_size
        self.bytes_sent = [0] * world_size

    def exchange_for(self, rank):
        def exchange(send, send_counts, recv_counts):
            assert sum(send_counts) == send.shape[0]
            self.bytes_sent[rank] += send.shape[0]
            self.chunks[rank] = send.split(send_counts)
            self.barrier.wait()
            received = [self.chunks[src][rank] for src in range(self.world_size)]
            self.barrier.wait()
            assert [chunk.shape[0] for chunk in received] == recv_counts
            return torch.cat(received)

        return exchange


def run_all_ranks(assignments, world_size):
    """Run every rank and return ({rank: (x, output)}, network)."""
    owner = contiguous_placement(SHAPE.num_experts, world_size)
    batch_size = assignments.shape[0]
    network = FakeNetwork(world_size)
    results, errors = {}, []

    def run_rank(rank):
        try:
            layer = MoELayer(SHAPE, owner, rank, 0, SEED)
            home = token_homes(batch_size, world_size) == rank
            x = torch.randn(int(home.sum()), SHAPE.hidden_size).to(torch.bfloat16)
            plan = build_plan(assignments, owner, world_size, rank)
            with torch.inference_mode():
                results[rank] = (x, layer(x, plan, exchange=network.exchange_for(rank)))
        except Exception as error:  # A thread's exception would otherwise be lost.
            errors.append(error)
            network.barrier.abort()

    threads = [threading.Thread(target=run_rank, args=(r,)) for r in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results, network


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 2, 7, 64])
@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_distributed_output_matches_reference(world_size, batch_size, alpha):
    generator = torch.Generator().manual_seed(batch_size)
    probabilities = zipf_probabilities(SHAPE.num_experts, alpha, SEED)
    assignments = sample_assignments(probabilities, batch_size, SHAPE.top_k, generator)

    results, _ = run_all_ranks(assignments, world_size)

    homes = token_homes(batch_size, world_size)
    for rank, (x, output) in results.items():
        assert output.shape == x.shape
        expected = reference_output(x, assignments[homes == rank], SHAPE, 0, SEED)
        assert outputs_match(output, expected)


def test_reference_check_detects_a_wrong_expert():
    assignments = torch.tensor([[0, 1, 2, 3]] * 8)
    results, _ = run_all_ranks(assignments, 2)
    x, output = results[0]
    assert outputs_match(output, reference_output(x, assignments[:4], SHAPE, 0, SEED))

    # Changing one of the four experts must be enough to fail the check.
    one_wrong = torch.tensor([[0, 1, 2, 8]] * 4)
    assert not outputs_match(output, reference_output(x, one_wrong, SHAPE, 0, SEED))
    # So must a mismatch in where tokens are sent.
    swapped = reference_output(x, assignments[:4], SHAPE, 0, SEED).flip(0)
    assert not outputs_match(output, swapped)


def test_plan_only_sends_remote_assignments():
    # 12 experts on 2 GPUs: 0-5 are on GPU 0, and 6-11 are on GPU 1.
    owner = contiguous_placement(12, 2)
    assignments = torch.tensor([
        [0, 1, 6, 7],  # home GPU 0: two local, two remote
        [6, 7, 8, 0],  # home GPU 1: three local, one remote
    ])
    plan_0 = build_plan(assignments, owner, 2, 0)
    plan_1 = build_plan(assignments, owner, 2, 1)

    assert plan_0.send_counts == [0, 2] and plan_0.recv_counts == [0, 1]
    assert plan_1.send_counts == [1, 0] and plan_1.recv_counts == [2, 0]
    assert plan_0.local_token_idx.tolist() == [0, 0]
    assert plan_1.local_token_idx.tolist() == [0, 0, 0]  # GPU 1's first token
    # Only experts owned by the rank get computed.
    assert {e for e, _ in plan_0.expert_groups} == {0, 1}
    assert {e for e, _ in plan_1.expert_groups} == {6, 7, 8}


def test_all_local_routing_skips_the_collective():
    assignments = torch.tensor([[0, 1, 2, 3], [6, 7, 8, 9]])
    results, network = run_all_ranks(assignments, 2)
    assert network.bytes_sent == [0, 0]
    assert build_plan(assignments, contiguous_placement(12, 2), 2, 0).any_remote is False
    assert results[0][1].abs().sum() > 0


def test_real_expert_size_matches_proposal():
    shape = ExpertShape()
    expert = Expert(shape, seed=0)
    size_mb = sum(p.numel() * 2 for p in expert.parameters()) / 1e6
    assert size_mb == pytest.approx(17.3, abs=0.1)
    x = torch.randn(3, shape.hidden_size).to(torch.bfloat16)
    assert expert(x).shape == x.shape
    assert expert(x).dtype == torch.bfloat16
