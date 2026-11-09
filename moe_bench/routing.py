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


class ZipfRouting:
    """Synthetic routing: each token picks top_k distinct experts with Zipf popularity."""

    def __init__(self, num_experts, top_k, alpha, seed):
        self.num_experts, self.top_k, self.alpha, self.seed = num_experts, top_k, alpha, seed
        self.label = "zipf"
        self.probabilities = zipf_probabilities(num_experts, alpha, seed)

    def load(self):
        return measured_load(self.probabilities, self.top_k, self.seed + 1000)

    def batches(self, batch_size, count, seed):
        generator = torch.Generator().manual_seed(seed)
        return [sample_assignments(self.probabilities, batch_size, self.top_k, generator) for _ in range(count)]


class TraceRouting:
    """Real routing recorded from a model: replays the experts each token actually chose.

    Each step takes batch_size consecutive recorded tokens from a random start, wrapping
    around at the end. Prefill tokens are in prompt order, so a step holds whole stretches
    of a few prompts. Decode tokens are reordered by decode step, then by prompt, so a step
    holds the next token of many prompts at once, as concurrent requests would.
    """

    def __init__(self, assignments, num_experts, layer, phase):
        self.assignments = assignments.long()
        self.num_experts, self.top_k = num_experts, assignments.shape[1]
        self.layer, self.phase = layer, phase
        self.alpha = float("nan")
        self.label = f"trace-{phase}-L{layer}"

    def load(self):
        """EPLB balances on the recorded load of the whole trace, as it would on past traffic."""
        return torch.bincount(self.assignments.flatten(), minlength=self.num_experts).double()

    def batches(self, batch_size, count, seed):
        generator = torch.Generator().manual_seed(seed)
        total = self.assignments.shape[0]
        starts = torch.randint(total, (count,), generator=generator)
        return [self.assignments[(start + torch.arange(batch_size)) % total] for start in starts.tolist()]


def load_trace(path, layer, phase):
    """Read one layer and phase of a trace written by model_trace.py, as a TraceRouting."""
    import json

    import numpy as np

    with np.load(path) as trace:
        meta = json.loads(str(trace["meta"]))
        if layer not in trace["layers"].tolist():
            raise ValueError(f"{path} has no MoE layer {layer}; it has {trace['layers'].tolist()}")
        assignments = trace[f"{phase}_{layer}"]
        if phase == "decode":
            order = np.lexsort((trace["decode_prompt"], trace["decode_step"]))
            assignments = assignments[order]
    return TraceRouting(torch.from_numpy(assignments.astype(np.int64)), meta["num_experts"], layer, phase), meta


def contiguous_placement(num_experts, world_size):
    """Return owner[expert]: the first experts on GPU 0, the next on GPU 1, and so on."""
    return torch.arange(num_experts) * world_size // num_experts


def measured_load(probabilities, top_k, seed, samples=100_000):
    """Count the assignments each expert receives, as EPLB's load statistics would.

    This differs from probabilities because a token picks an expert at most once,
    which caps the hottest experts. At alpha=1.5 the hottest has p=0.43 but receives
    23% of assignments, and balancing on p leaves GPU imbalance near 1.3.
    """
    generator = torch.Generator().manual_seed(seed)
    assignments = sample_assignments(probabilities, samples, top_k, generator)
    return torch.bincount(assignments.flatten(), minlength=probabilities.numel()).double()


def eplb_placement(load, num_physical, world_size):
    """Return (physical_to_logical, owner) from DeepSeek EPLB's policy on one node.

    Follows replicate_experts and balanced_packing in github.com/deepseek-ai/EPLB
    (MIT licence). Each extra copy goes to the expert with the highest load per copy,
    then copies are packed onto GPUs, heaviest first, into the lightest GPU with room.
    Copies of one expert can share a GPU, and each is then a separate activation.
    """
    if num_physical % world_size != 0:
        raise ValueError(f"{num_physical} physical experts don't divide across {world_size} GPUs")
    load = load.float()
    counts = torch.ones_like(load, dtype=torch.int64)
    copy_of = list(range(load.numel()))
    for _ in range(num_physical - load.numel()):
        hottest = (load / counts).max(dim=-1).indices
        copy_of.append(int(hottest))
        counts[hottest] += 1
    copy_of = torch.tensor(copy_of)
    load_per_copy = (load / counts)[copy_of]

    per_gpu = num_physical // world_size
    gpu_load = [0.0] * world_size
    gpu_items = [0] * world_size
    physical_to_logical = torch.empty(num_physical, dtype=torch.int64)
    for copy in load_per_copy.sort(descending=True).indices.tolist():
        gpu = min((g for g in range(world_size) if gpu_items[g] < per_gpu), key=gpu_load.__getitem__)
        physical_to_logical[gpu * per_gpu + gpu_items[gpu]] = copy_of[copy]
        gpu_load[gpu] += load_per_copy[copy].item()
        gpu_items[gpu] += 1
    return physical_to_logical, contiguous_placement(num_physical, world_size)


def route_to_replicas(assignments, physical_to_logical, generator):
    """Turn logical expert IDs into physical ones, splitting each expert's tokens evenly.

    Each expert's assignments go round-robin across its copies. The first copy is
    random, because a fixed one would send small batches to the same GPU every step.
    Every rank passes the same generator state, so they all get the same split.
    """
    flat = assignments.flatten()
    physical = torch.empty_like(flat)
    for expert in flat.unique():
        mask = flat == expert
        copies = (physical_to_logical == expert).nonzero().flatten()
        first = torch.randint(copies.numel(), (1,), generator=generator)
        physical[mask] = copies[(first + torch.arange(int(mask.sum()))) % copies.numel()]
    return physical.view(assignments.shape)


@dataclass
class LoadMetrics:
    expert_imbalance: float
    gpu_imbalance: float
    max_tokens_per_gpu: int
    max_activated_per_gpu: int
    tokens_per_gpu: list
    activated_per_gpu: list


def load_metrics(assignments, owner, world_size, physical_to_logical=None):
    """Compute the imbalance metrics defined in the proposal for one batch.

    assignments holds physical expert IDs, so activated counts copies. Expert
    imbalance is over logical experts, because replicating doesn't change demand.
    """
    if physical_to_logical is None:
        physical_to_logical = torch.arange(owner.numel())
    flat = assignments.flatten()
    expert_load = torch.bincount(physical_to_logical[flat], minlength=int(physical_to_logical.max()) + 1)
    copy_load = torch.bincount(flat, minlength=owner.numel())
    gpu_load = torch.bincount(owner[flat], minlength=world_size)
    activated = torch.bincount(owner[copy_load > 0], minlength=world_size)
    return LoadMetrics(
        expert_imbalance=(expert_load.max() / expert_load.float().mean()).item(),
        gpu_imbalance=(gpu_load.max() / gpu_load.float().mean()).item(),
        max_tokens_per_gpu=gpu_load.max().item(),
        max_activated_per_gpu=activated.max().item(),
        tokens_per_gpu=gpu_load.tolist(),
        activated_per_gpu=activated.tolist(),
    )
