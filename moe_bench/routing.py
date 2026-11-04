"""Synthetic expert routing, placement, and the load metrics derived from them."""
from dataclasses import dataclass

import torch


def zipf_probabilities(num_experts, alpha, seed):
    """Return p(expert) with p proportional to rank^-alpha.

    Popularity ranks are shuffled across expert IDs with a fixed seed. Without
    the shuffle the hottest experts would be 0, 1, 2, ..., which all sit on
    GPU 0 under contiguous placement, making "contiguous" the same as the
    concentrated-hot placement that Q4 studies separately.
    """
    ranks = torch.arange(1, num_experts + 1, dtype=torch.float64)
    by_rank = ranks.pow(-alpha)
    by_rank /= by_rank.sum()
    generator = torch.Generator().manual_seed(seed)
    popularity_order = torch.randperm(num_experts, generator=generator)
    probabilities = torch.empty(num_experts, dtype=torch.float64)
    probabilities[popularity_order] = by_rank
    return probabilities


def sample_assignments(probabilities, batch_size, top_k, generator):
    """Return a (batch_size, top_k) tensor of expert IDs, distinct within a token."""
    return torch.multinomial(
        probabilities.expand(batch_size, -1), top_k, replacement=False, generator=generator
    )


def contiguous_placement(num_experts, world_size):
    """Return owner[expert]: the first experts on GPU 0, the next on GPU 1, and so on."""
    return torch.arange(num_experts) * world_size // num_experts


@dataclass
class LoadMetrics:
    expert_imbalance: float
    gpu_imbalance: float
    max_tokens_per_gpu: int
    max_activated_per_gpu: int
    tokens_per_gpu: list
    activated_per_gpu: list


def load_metrics(assignments, owner, world_size):
    """Compute the imbalance metrics defined in the proposal for one batch."""
    expert_load = torch.bincount(assignments.flatten(), minlength=owner.numel())
    gpu_load = torch.bincount(owner[assignments.flatten()], minlength=world_size)
    activated = torch.bincount(owner[expert_load > 0], minlength=world_size)
    return LoadMetrics(
        expert_imbalance=(expert_load.max() / expert_load.float().mean()).item(),
        gpu_imbalance=(gpu_load.max() / gpu_load.float().mean()).item(),
        max_tokens_per_gpu=gpu_load.max().item(),
        max_activated_per_gpu=activated.max().item(),
        tokens_per_gpu=gpu_load.tolist(),
        activated_per_gpu=activated.tolist(),
    )
