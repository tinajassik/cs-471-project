"""Expert-parallel MoE layer: dispatch tokens to experts, compute, combine results."""
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ExpertShape:
    """Defaults match Qwen1.5-MoE-A2.7B."""

    num_experts: int = 60
    top_k: int = 4
    hidden_size: int = 2048
    expert_size: int = 1408


class Expert(nn.Module):
    """A Qwen2-MoE expert: down(silu(gate(x)) * up(x)), about 17 MB in bf16."""

    def __init__(self, shape, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)

        def weight(rows, columns):
            values = 0.02 * torch.randn(rows, columns, generator=generator)
            return nn.Parameter(values.to(torch.bfloat16), requires_grad=False)

        self.gate = weight(shape.expert_size, shape.hidden_size)
        self.up = weight(shape.expert_size, shape.hidden_size)
        self.down = weight(shape.hidden_size, shape.expert_size)

    def forward(self, x):
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)


def expert_seed(seed, layer, expert):
    """Give each expert fixed weights, so any rank can rebuild any expert."""
    return seed + 1_000_000 * (layer + 1) + expert


@dataclass
class StepPlan:
    """Everything one rank needs for one step, so no routing work is timed."""

    send_token_idx: torch.Tensor  # home tokens to send, ordered by destination
    send_counts: list
    recv_counts: list
    local_token_idx: torch.Tensor  # home tokens whose expert is on this rank
    expert_groups: list  # (physical expert ID, positions in the inbox)
    any_remote: bool  # whether any rank sends anything this step

    def to(self, device):
        self.send_token_idx = self.send_token_idx.to(device)
        self.local_token_idx = self.local_token_idx.to(device)
        self.expert_groups = [(e, p.to(device)) for e, p in self.expert_groups]
        return self


def token_homes(batch_size, world_size):
    """Return the rank each token starts on, split as evenly as possible."""
    return torch.arange(batch_size) * world_size // batch_size


def build_plan(assignments, owner, world_size, rank):
    """Work out this rank's traffic and compute from the shared routing trace.

    Only remote assignments enter the collective, as in the proposal: bytes
    moved depend on routing and placement. A token selecting two experts on one
    GPU is sent twice. Every rank walks the assignments in the same order, so
    each receiver knows which expert each incoming token is for. assignments and
    owner use physical expert IDs, so a replica is treated like any other expert.
    """
    batch_size, top_k = assignments.shape
    homes = token_homes(batch_size, world_size)
    token = torch.arange(batch_size).repeat_interleave(top_k)
    expert = assignments.flatten()
    src = homes[token]
    dst = owner[expert]
    # x holds only this rank's tokens, so index it from this rank's first token.
    position = token - (homes < rank).sum()

    local = (src == rank) & (dst == rank)
    send_tokens, send_counts = [], []
    recv_experts, recv_counts = [], []
    for peer in range(world_size):
        if peer == rank:
            send_mask = recv_mask = torch.zeros_like(token, dtype=torch.bool)
        else:
            send_mask = (src == rank) & (dst == peer)
            recv_mask = (src == peer) & (dst == rank)
        send_tokens.append(position[send_mask])
        send_counts.append(int(send_mask.sum()))
        recv_experts.append(expert[recv_mask])
        recv_counts.append(int(recv_mask.sum()))

    inbox_experts = torch.cat([expert[local], *recv_experts])
    expert_groups = [
        (int(e), (inbox_experts == e).nonzero().squeeze(1)) for e in inbox_experts.unique()
    ]
    return StepPlan(
        send_token_idx=torch.cat(send_tokens),
        send_counts=send_counts,
        recv_counts=recv_counts,
        local_token_idx=position[local],
        expert_groups=expert_groups,
        any_remote=bool((src != dst).any()),
    )


def all_to_all_exchange(send, send_counts, recv_counts):
    """Move tokens between GPUs with NCCL. Other exchanges are used only in tests."""
    recv = send.new_empty(sum(recv_counts), send.shape[1])
    dist.all_to_all_single(recv, send, recv_counts, send_counts)
    return recv


class Phases:
    """Default no-op phase markers. benchmark.py swaps in timed ones."""

    def mark(self, name):
        pass


class MoELayer(nn.Module):
    """The physical experts this rank owns. Each copy has its own weights in memory."""

    def __init__(self, shape, owner, rank, layer_index, seed, physical_to_logical=None):
        super().__init__()
        self.top_k = shape.top_k
        if physical_to_logical is None:
            physical_to_logical = torch.arange(owner.numel())
        self.experts = nn.ModuleDict({
            str(e): Expert(shape, expert_seed(seed, layer_index, int(physical_to_logical[e])))
            for e in (owner == rank).nonzero().flatten().tolist()
        })

    def forward(self, x, plan, exchange=all_to_all_exchange, phases=Phases()):
        """Return the combined expert output for this rank's home tokens."""
        phases.mark("start")

        # Dispatch: send each remote assignment to the GPU holding its expert.
        send = x[plan.send_token_idx]
        if plan.any_remote:
            received = exchange(send, plan.send_counts, plan.recv_counts)
        else:
            received = send
        inbox = torch.cat([x[plan.local_token_idx], received])
        phases.mark("dispatch")

        # Compute: run each activated expert once, on all of its tokens.
        results = inbox.new_empty(inbox.shape)
        for expert_id, positions in plan.expert_groups:
            results[positions] = self.experts[str(expert_id)](inbox[positions])
        phases.mark("compute")

        # Combine: return results to their home GPUs and average per token.
        num_local = plan.local_token_idx.numel()
        if plan.any_remote:
            returned = exchange(results[num_local:], plan.recv_counts, plan.send_counts)
        else:
            returned = results[num_local:]
        output = torch.zeros_like(x)
        output.index_add_(0, plan.local_token_idx, results[:num_local] / self.top_k)
        output.index_add_(0, plan.send_token_idx, returned / self.top_k)
        phases.mark("combine")
        return output


@lru_cache(maxsize=None)
def _reference_expert(shape, seed, layer_index, expert, device):
    return Expert(shape, expert_seed(seed, layer_index, expert)).to(device)


def reference_output(x, assignments, shape, layer_index, seed):
    """Compute the expected output without dispatch, to check the distributed path."""
    output = torch.zeros_like(x)
    for e in assignments.unique().tolist():
        tokens = (assignments == e).any(dim=1).nonzero().flatten()
        expert = _reference_expert(shape, seed, layer_index, e, x.device)
        output[tokens] += expert(x[tokens]) / shape.top_k
    return output


def outputs_match(output, expected, tolerance=0.05):
    """Compare relative to the output's size, leaving room for bf16 rounding."""
    error = (output.float() - expected.float()).norm()
    return bool(error <= tolerance * expected.float().norm())
