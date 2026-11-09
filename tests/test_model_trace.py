"""Runs the tracer on tiny random Qwen MoE models on the CPU. Skipped without transformers."""
import json

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from moe_bench import model_trace  # noqa: E402
from moe_bench.routing import load_trace  # noqa: E402

TINY = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
            max_position_embeddings=128)


def tiny_model(kind):
    torch.manual_seed(0)
    if kind == "qwen2_moe":
        config = transformers.Qwen2MoeConfig(**TINY, shared_expert_intermediate_size=32)
        return transformers.Qwen2MoeForCausalLM(config).eval()
    config = transformers.Qwen3MoeConfig(**TINY)
    return transformers.Qwen3MoeForCausalLM(config).eval()


@pytest.fixture(params=["qwen2_moe", "qwen3_moe"])
def model(request):
    return tiny_model(request.param)


def hooks(model):
    blocks = model_trace.moe_blocks(model)
    return blocks, model_trace.RoutingRecorder(blocks, 2), model_trace.BlockTimer(blocks, cuda=False)


def test_every_layer_is_an_moe_block(model):
    assert [index for index, _, _ in model_trace.moe_blocks(model)] == [0, 1]


def test_recorded_routing_matches_the_model_router(model):
    _, recorder, _ = hooks(model)
    ids = torch.randint(64, (1, 7), generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        output = model(input_ids=ids, output_router_logits=True)
    chosen = recorder.collect()
    for layer, logits in enumerate(output.router_logits):
        expected = logits.float().softmax(-1).topk(2, dim=-1).indices
        assert chosen[layer].shape == (7, 2)
        assert (np.sort(chosen[layer], axis=1) == np.sort(expected.numpy(), axis=1)).all()


def test_hooks_do_not_change_the_output(model):
    ids = torch.randint(64, (1, 5), generator=torch.Generator().manual_seed(2))
    with torch.inference_mode():
        before = model(input_ids=ids).logits
        hooks(model)
        after = model(input_ids=ids).logits
    assert torch.equal(before, after)


def test_prefill_then_greedy_decode(model):
    _, recorder, timer = hooks(model)
    ids = torch.randint(64, (1, 6), generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        records = model_trace.run_prompt(model, ids, 4, recorder, timer)
    assert [(r["phase"], r["step"], r["tokens"], r["context"]) for r in records] == [
        ("prefill", 0, 6, 6), ("decode", 1, 1, 7), ("decode", 2, 1, 8), ("decode", 3, 1, 9), ("decode", 4, 1, 10),
    ]
    assert records[0]["routing"][0].shape == (6, 2)
    assert records[1]["routing"][1].shape == (1, 2)
    for r in records:
        assert r["attention_ms"] > 0 and r["moe_ms"] > 0
        assert r["total_ms"] >= r["attention_ms"] + r["moe_ms"]


def test_decode_matches_generate(model):
    _, recorder, timer = hooks(model)
    ids = torch.randint(64, (1, 6), generator=torch.Generator().manual_seed(4))
    with torch.inference_mode():
        expected = model.generate(ids, max_new_tokens=4, do_sample=False)[0, 6:]
        records = model_trace.run_prompt(model, ids, 3, recorder, timer)
    # The prefill chooses the first new token, and each decode step the next one.
    assert [r["next_token"] for r in records] == expected.tolist()


def test_stops_at_end_of_text(model):
    _, recorder, timer = hooks(model)
    ids = torch.randint(64, (1, 4), generator=torch.Generator().manual_seed(5))
    with torch.inference_mode():
        first = int(model(input_ids=ids).logits[0, -1].argmax())
        records = model_trace.run_prompt(model, ids, 10, recorder, timer, eos_ids={first})
    assert len(records) == 1


def test_disabled_hooks_record_nothing(model):
    _, recorder, timer = hooks(model)
    recorder.enabled = timer.enabled = False
    with torch.inference_mode():
        model(input_ids=torch.zeros(1, 3, dtype=torch.long))
    assert recorder.collect() == {}
    assert timer.collect() == {"attention": 0.0, "moe": 0.0}


def test_saved_trace_replays_in_the_benchmark(model, tmp_path):
    _, recorder, timer = hooks(model)
    writer = model_trace.TraceWriter([0, 1])
    with torch.inference_mode():
        for index, length in enumerate((5, 3)):
            ids = torch.randint(64, (1, length), generator=torch.Generator().manual_seed(index))
            writer.add(index, ["gsm8k", "humaneval"][index], model_trace.run_prompt(model, ids, 2, recorder, timer))
    meta = {"model": "tiny", "num_experts": 8, "top_k": 2, "hidden_size": 32, "moe_intermediate_size": 16}
    output = tmp_path / "trace_tiny1.5"
    writer.save(output, meta, ["gsm8k", "humaneval"], 2)

    saved = np.load(tmp_path / "trace_tiny1.5.npz")
    assert saved["prefill_0"].shape == (8, 2) and saved["decode_1"].shape == (4, 2)
    assert saved["prefill_prompt"].tolist() == [0] * 5 + [1] * 3
    assert saved["decode_step"].tolist() == [1, 2, 1, 2]
    assert json.loads(str(saved["meta"]))["top_k"] == 2
    timing = (tmp_path / "trace_tiny1.5_timing.csv").read_text().splitlines()
    assert timing[0].split(",") == model_trace.TIMING_FIELDS and len(timing) == 1 + 6

    routing, loaded_meta = load_trace(tmp_path / "trace_tiny1.5.npz", 1, "decode")
    # Decode tokens come back ordered by step, then prompt: step 1 of both prompts first.
    order = np.lexsort((saved["decode_prompt"], saved["decode_step"]))
    assert routing.assignments.tolist() == saved["decode_1"][order].tolist()
    assert loaded_meta == meta


def test_end_tokens_come_from_the_tokenizer_and_the_generation_config(model):
    class Tokenizer:
        eos_token_id = 5

    model.generation_config.eos_token_id = [6, 7]
    assert model_trace.end_tokens(Tokenizer(), model) == {5, 6, 7}
    model.generation_config.eos_token_id = None
    assert model_trace.end_tokens(Tokenizer(), model) == {5}
