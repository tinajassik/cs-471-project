#!/bin/bash -l
#SBATCH --job-name=model_trace
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out

# Q5: record the real routing of each model, and time its attention and MoE blocks, on
# one GPU with the unmodified Hugging Face model. On transformers 5.x, which runs all
# experts as one grouped GEMM by default, a shorter second pass uses the per-expert
# loop (eager), to compare the two on the real model. Run jobs/fetch_inputs.sh first.
# Submit from the repo root: sbatch jobs/model_trace.sbatch
# One model only: sbatch --export=ALL,MODELS=Qwen3-30B-A3B jobs/model_trace.sbatch

source jobs/env.sh
export HF_HUB_OFFLINE=1
mkdir -p results

MODELS=${MODELS:-"Qwen1.5-MoE-A2.7B Qwen3-30B-A3B"}
MAJOR=$(python -c "import transformers; print(transformers.__version__.split('.')[0])")
nvidia-smi --query-gpu=name,memory.total --format=csv

for MODEL in $MODELS; do
    TAG=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
    NAME="results/trace_${TAG}_${SLURM_JOB_ID}"
    python -m moe_bench.model_trace --model "$GROUP_DIR/models/$MODEL" --prompts "$GROUP_DIR/prompts.jsonl" \
        --output "$NAME"
    python analysis/analyse_trace.py "$NAME.npz"

    if [ "$MAJOR" -ge 5 ]; then
        python -m moe_bench.model_trace --model "$GROUP_DIR/models/$MODEL" --prompts "$GROUP_DIR/prompts.jsonl" \
            --output "${NAME}_eager" --experts-implementation eager --max-prompts 40
        python analysis/analyse_trace.py "${NAME}_eager.npz" | sed -n '/Time per forward/,$p'
    fi
done
