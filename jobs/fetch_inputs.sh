#!/bin/bash -l
# Downloads both models and the prompts for the real-model jobs. Compute nodes have no
# internet, so run this once on the login node, from the repo root:
#     bash jobs/fetch_inputs.sh
# Needs about 90 GB in $GROUP_DIR/models (28.6 GB + 61 GB of bf16 weights).

set -euo pipefail
source jobs/env.sh

# transformers 4.51 is the first with Qwen3-MoE; accelerate loads weights straight onto
# the GPU; datasets fetches the prompts.
python - <<'EOF' || pip install "transformers>=4.51" accelerate datasets
import importlib.util, sys
import transformers
missing = [m for m in ("accelerate", "datasets") if importlib.util.find_spec(m) is None]
major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
sys.exit(1 if missing or (major, minor) < (4, 51) else 0)
EOF

mkdir -p "$GROUP_DIR/models"
df -h "$GROUP_DIR"
for MODEL in Qwen/Qwen1.5-MoE-A2.7B Qwen/Qwen3-30B-A3B; do
    python -c "from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" \
        "$MODEL" "$GROUP_DIR/models/${MODEL#*/}"
done

python -m moe_bench.prompts --output "$GROUP_DIR/prompts.jsonl" --per-source 150
python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
