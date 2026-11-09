"""Write the prompts for model_trace.py: GSM8K questions and HumanEval code prompts.

Needs internet and the datasets package, so run it on the login node:
    python -m moe_bench.prompts --output $GROUP_DIR/prompts.jsonl --per-source 150

Each line is {"id", "source", "text"}. Two workloads, maths and code, so routing can
be compared across them, as planned in the proposal.
"""
import argparse
import json
from pathlib import Path

SOURCES = {
    # source name: (dataset, config, split, text field)
    "gsm8k": ("openai/gsm8k", "main", "test", "question"),
    "humaneval": ("openai/openai_humaneval", None, "test", "prompt"),
}


def interleave(groups):
    """Alternate sources, so any prefix of the file holds both workloads."""
    out = []
    for row in zip(*groups):
        out.extend(row)
    return out


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=150)
    args = parser.parse_args()

    groups = []
    for source, (name, config, split, field) in SOURCES.items():
        rows = load_dataset(name, config, split=split).select(range(args.per_source))
        groups.append([{"id": f"{source}-{i}", "source": source, "text": row[field]} for i, row in enumerate(rows)])
    prompts = interleave(groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as file:
        file.writelines(json.dumps(p) + "\n" for p in prompts)
    print(f"wrote {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
