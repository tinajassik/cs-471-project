#!/bin/bash -l
# Downloads models from Hugging Face (huggingface.co/Qwen/...) into $GROUP_DIR/models, and
# the prompts into $GROUP_DIR/prompts.jsonl. Compute nodes have no internet, so run this
# on the login node, from the repo root:
#     bash jobs/fetch_inputs.sh
# One model only (Qwen3-30B-A3B is 61 GB, Qwen1.5-MoE-A2.7B 28.6 GB):
#     MODELS=Qwen/Qwen3-30B-A3B bash jobs/fetch_inputs.sh
# The class shares its storage, so delete a model's folder once its trace is done:
# the trace files in results/ are all the replay needs.

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

MODELS=${MODELS:-"Qwen/Qwen1.5-MoE-A2.7B Qwen/Qwen3-30B-A3B"}
mkdir -p "$GROUP_DIR/models"
for MODEL in $MODELS; do
    # Only what loading needs: the weights, configs and tokenizer, not other formats.
    python -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model', '*.tiktoken'])
" "$MODEL" "$GROUP_DIR/models/${MODEL#*/}"
done
du -sh "$GROUP_DIR"/models/*

python -m moe_bench.prompts --output "$GROUP_DIR/prompts.jsonl" --per-source 150
python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
