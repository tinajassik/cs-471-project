"""Run a real MoE model on prompts, record its routing, and time its attention and MoE blocks.

Runs the unmodified Hugging Face model on one GPU, one prompt at a time, with greedy
decoding. A forward hook on each MoE block's router records which experts every token
selects; hooks only observe, so the model's output is unchanged. Other hooks time each
layer's attention and MoE block with CUDA events and label them for Nsight Systems.

    python -m moe_bench.model_trace --model $GROUP_DIR/models/Qwen1.5-MoE-A2.7B \
        --prompts $GROUP_DIR/prompts.jsonl --output results/trace_qwen1.5

Writes <output>.npz (routing, read by analysis/analyse_trace.py and by
benchmark.py --trace) and <output>_timing.csv (one row per forward pass).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

TIMING_FIELDS = [
    "prompt", "source", "phase", "step", "tokens", "context",
    "total_ms", "attention_ms", "moe_ms", "other_ms",
]


def moe_blocks(model):
    """Return [(layer index, decoder layer, MoE block)] for every layer whose MLP is routed."""
    blocks = [
        (index, layer, layer.mlp)
        for index, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts")
    ]
    if not blocks:
        raise ValueError(f"{type(model).__name__} has no MoE blocks with a gate and experts")
    return blocks


def selected_experts(router_output, top_k):
    """Return the (tokens, top_k) expert IDs a router chose.

    In transformers 4.x the router is a Linear that returns logits, and the block then
    takes softmax and top-k; softmax keeps the order, so top-k of the logits is the same
    choice. In 5.x the router returns (logits, scores, indices).
    """
    if isinstance(router_output, tuple):
        for item in router_output:
            if torch.is_tensor(item) and not item.is_floating_point():
                return item.reshape(-1, top_k)
        router_output = router_output[0]
    return router_output.reshape(-1, router_output.shape[-1]).topk(top_k, dim=-1).indices


class RoutingRecorder:
    """Keeps each router's choices on the GPU during a forward pass, so it adds no sync."""

    def __init__(self, blocks, top_k):
        self.top_k = top_k
        self.enabled = True
        self.pending = {index: [] for index, _, _ in blocks}
        self.handles = [block.gate.register_forward_hook(self._hook(index)) for index, _, block in blocks]

    def _hook(self, index):
        def hook(module, inputs, output):
            if self.enabled:
                self.pending[index].append(selected_experts(output, self.top_k).to(torch.uint8))
        return hook

    def collect(self):
        """Return {layer: (tokens, top_k) uint8 array} for the last forward pass."""
        chosen = {index: torch.cat(parts).cpu().numpy() for index, parts in self.pending.items() if parts}
        for parts in self.pending.values():
            parts.clear()
        return chosen


class WallClockEvent:
    """Stands in for a CUDA event on the CPU, so the same code runs in tests."""

    def record(self):
        self.time = time.perf_counter()

    def elapsed_time(self, end):
        return (end.time - self.time) * 1000


class BlockTimer:
    """Times every layer's attention and MoE block, and labels them with NVTX on the GPU."""

    def __init__(self, blocks, cuda):
        self.cuda = cuda
        self.enabled = True
        self.spans = []
        self.open = {}
        self.handles = []
        for index, layer, block in blocks:
            for name, module in (("attention", layer.self_attn), ("moe", block)):
                self.handles.append(module.register_forward_pre_hook(self._start(name, index)))
                self.handles.append(module.register_forward_hook(self._stop(name, index)))

    def event(self):
        return torch.cuda.Event(enable_timing=True) if self.cuda else WallClockEvent()

    def _start(self, name, index):
        def hook(module, inputs):
            if self.enabled:
                if self.cuda:
                    torch.cuda.nvtx.range_push(f"{name} {index}")
                self.open[name, index] = self.event()
                self.open[name, index].record()
        return hook

    def _stop(self, name, index):
        def hook(module, inputs, output):
            if self.enabled:
                end = self.event()
                end.record()
                self.spans.append((name, self.open.pop((name, index)), end))
                if self.cuda:
                    torch.cuda.nvtx.range_pop()
        return hook

    def collect(self):
        """Return total ms per block kind for the last forward pass. Call after a sync."""
        totals = {"attention": 0.0, "moe": 0.0}
        for name, start, end in self.spans:
            totals[name] += start.elapsed_time(end)
        self.spans.clear()
        return totals


def run_prompt(model, input_ids, new_tokens, recorder, timer, eos_ids=()):
    """Prefill the prompt, then decode greedily. Returns one record per forward pass.

    Each record holds the phase, the decode step (0 for prefill), the tokens it
    processed, the context length, the routing of those tokens, the block timings and
    the token it chose next.
    Decoding stops after new_tokens tokens or at an end-of-text token.
    """
    records = []
    past = None
    ids = input_ids
    context = 0
    for step in range(new_tokens + 1):
        if timer.cuda:
            torch.cuda.nvtx.range_push("prefill" if step == 0 else "decode")
        start, end = timer.event(), timer.event()
        start.record()
        output = model(input_ids=ids, past_key_values=past, use_cache=True)
        end.record()
        if timer.cuda:
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
        context += ids.shape[1]
        blocks = timer.collect()
        total = start.elapsed_time(end)
        records.append({
            "phase": "prefill" if step == 0 else "decode",
            "step": step,
            "tokens": ids.shape[1],
            "context": context,
            "routing": recorder.collect(),
            "total_ms": total,
            "attention_ms": blocks["attention"],
            "moe_ms": blocks["moe"],
            "other_ms": total - blocks["attention"] - blocks["moe"],
        })
        past = output.past_key_values
        ids = output.logits[:, -1:].argmax(dim=-1)
        records[-1]["next_token"] = int(ids)
        if int(ids) in eos_ids:
            break
    return records


class TraceWriter:
    """Collects routing per layer and phase across prompts, and timing rows."""

    def __init__(self, layers):
        self.layers = layers
        self.routing = {(phase, layer): [] for phase in ("prefill", "decode") for layer in layers}
        self.positions = {"prefill_prompt": [], "decode_prompt": [], "decode_step": []}
        self.timing = []

    def add(self, prompt_index, source, records):
        for record in records:
            phase = record["phase"]
            for layer in self.layers:
                self.routing[phase, layer].append(record["routing"][layer])
            self.positions[f"{phase}_prompt"].append(np.full(record["tokens"], prompt_index, dtype=np.int32))
            if phase == "decode":
                self.positions["decode_step"].append(np.full(record["tokens"], record["step"], dtype=np.int32))
            self.timing.append({
                "prompt": prompt_index, "source": source,
                **{field: record[field] for field in TIMING_FIELDS[2:]},
            })

    def save(self, output, meta, sources, top_k):
        arrays = {"layers": np.array(self.layers), "meta": np.array(json.dumps(meta)),
                  "prompt_source": np.array(sources)}
        for (phase, layer), parts in self.routing.items():
            arrays[f"{phase}_{layer}"] = np.concatenate(parts) if parts else np.empty((0, top_k), np.uint8)
        for name, parts in self.positions.items():
            arrays[name] = np.concatenate(parts) if parts else np.empty(0, np.int32)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path(output), **arrays)
        with open(output.parent / f"{output.name}_timing.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, TIMING_FIELDS)
            writer.writeheader()
            writer.writerows(self.timing)


def trace_path(output):
    """Add .npz without with_suffix, which would turn trace_qwen1.5 into trace_qwen1.npz."""
    return output.parent / f"{output.name}.npz"


def load_model(path, experts_implementation, device):
    """Load bf16 weights, straight onto the device when accelerate is installed.

    transformers 4.x calls dtype torch_dtype. Without accelerate the weights load on the
    CPU first, which needs as much host memory as the model (61 GB for Qwen3-30B-A3B).
    """
    import importlib.util

    import transformers

    major = int(transformers.__version__.split(".")[0])
    kwargs = {"dtype" if major >= 5 else "torch_dtype": torch.bfloat16}
    if experts_implementation:
        kwargs["experts_implementation"] = experts_implementation
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    if has_accelerate:
        kwargs["device_map"] = device
    model = transformers.AutoModelForCausalLM.from_pretrained(path, **kwargs)
    return (model if has_accelerate else model.to(device)).eval()


def model_meta(model):
    """The shape benchmark.py needs to replay this model's routing, plus versions."""
    import transformers

    config = model.config
    return {
        "model": config.name_or_path,
        "num_experts": config.num_experts,
        "top_k": config.num_experts_per_tok,
        "hidden_size": config.hidden_size,
        "moe_intermediate_size": config.moe_intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "experts_implementation": getattr(config, "_experts_implementation", None),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }


def end_tokens(tokenizer, model):
    """End-of-text IDs from the tokenizer and the generation config, which can list several."""
    ids = {tokenizer.eos_token_id}
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids.update(configured if isinstance(configured, list) else [configured])
    return {i for i in ids if i is not None}


def read_prompts(path, limit):
    with open(path) as file:
        prompts = [json.loads(line) for line in file if line.strip()]
    return prompts[:limit] if limit else prompts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="local model directory or Hugging Face ID")
    parser.add_argument("--prompts", type=Path, required=True,
                        help="JSON lines with 'text' and 'source', from moe_bench.prompts")
    parser.add_argument("--output", type=Path, required=True, help="output prefix, without extension")
    parser.add_argument("--max-prompts", type=int, default=0, help="0 uses every prompt")
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=128, help="decode steps per prompt at most")
    parser.add_argument("--warmup-prompts", type=int, default=2,
                        help="prompts run first and not recorded, so start-up costs stay out of the timings")
    parser.add_argument("--experts-implementation", default=None,
                        help="transformers 5.x only: eager (a loop with launches per expert) or grouped_mm "
                             "(one grouped GEMM, the 5.x default)")
    parser.add_argument("--no-block-timing", action="store_true",
                        help="skip the per-block CUDA events, which cost a few us of CPU time per block")
    return parser.parse_args()


def main():
    from transformers import AutoTokenizer

    args = parse_args()
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.experts_implementation, device)

    blocks = moe_blocks(model)
    layers = [index for index, _, _ in blocks]
    meta = model_meta(model)
    recorder = RoutingRecorder(blocks, meta["top_k"])
    timer = BlockTimer(blocks, cuda)
    eos_ids = end_tokens(tokenizer, model)
    print(f"{meta['model']} on {meta['device']}: {len(layers)} MoE layers, {meta['num_experts']} experts, "
          f"top-{meta['top_k']}, experts implementation {meta['experts_implementation']}", flush=True)

    prompts = read_prompts(args.prompts, args.max_prompts)
    writer = TraceWriter(layers)

    def encode(text):
        ids = tokenizer(text, return_tensors="pt").input_ids[:, :args.max_prompt_tokens]
        return ids.to(device)

    with torch.inference_mode():
        recorder.enabled = timer.enabled = False
        for prompt in prompts[:args.warmup_prompts]:
            run_prompt(model, encode(prompt["text"]), args.new_tokens, recorder, timer, eos_ids)
        recorder.enabled, timer.enabled = True, not args.no_block_timing
        for index, prompt in enumerate(prompts):
            records = run_prompt(model, encode(prompt["text"]), args.new_tokens, recorder, timer, eos_ids)
            writer.add(index, prompt["source"], records)
            if index % 25 == 0:
                decode_ms = [r["total_ms"] for r in records[1:]]
                print(f"prompt {index}/{len(prompts)}: {records[0]['tokens']} prompt tokens, prefill "
                      f"{records[0]['total_ms']:.1f} ms, {len(decode_ms)} decode steps, median "
                      f"{np.median(decode_ms) if decode_ms else float('nan'):.1f} ms", flush=True)

    writer.save(args.output, meta, [p["source"] for p in prompts], meta["top_k"])
    print(f"TRACE FINISHED, wrote {trace_path(args.output)} and {args.output.name}_timing.csv", flush=True)


if __name__ == "__main__":
    main()
