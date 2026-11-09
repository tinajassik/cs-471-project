#!/bin/bash -l
#SBATCH --job-name=model_timeline
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

# Q5: is the real model's MoE block waiting on the CPU (kernel launches) or on the GPU?
# Traces a few prompts with Nsight Systems, with the grouped GEMM and with the per-expert
# loop, then reports how busy the GPU was inside the MoE blocks. Tracing slows launches,
# so use these runs for that question only, not for timings.
# Submit from the repo root: sbatch jobs/model_timeline.sbatch
# Other model: sbatch --export=ALL,MODEL=Qwen1.5-MoE-A2.7B jobs/model_timeline.sbatch

module purge
module load gcc cuda
module load nvhpc
NSYS=$(command -v nsys)

source jobs/env.sh
export HF_HUB_OFFLINE=1
mkdir -p results

MODEL=${MODEL:-Qwen3-30B-A3B}
TAG=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
# transformers 4.x has only its per-expert loop, and no option to choose.
MAJOR=$(python -c "import transformers; print(transformers.__version__.split('.')[0])")
if [ "$MAJOR" -ge 5 ]; then IMPLEMENTATIONS="grouped_mm eager"; else IMPLEMENTATIONS="default"; fi

for IMPLEMENTATION in $IMPLEMENTATIONS; do
    NAME="results/model_timeline_${TAG}_${IMPLEMENTATION}_${SLURM_JOB_ID}"
    CHOICE=()
    if [ "$IMPLEMENTATION" != default ]; then CHOICE=(--experts-implementation "$IMPLEMENTATION"); fi
    "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true -o "$NAME" \
        python -m moe_bench.model_trace --model "$GROUP_DIR/models/$MODEL" --prompts "$GROUP_DIR/prompts.jsonl" \
        --output "$NAME" "${CHOICE[@]}" --max-prompts 4 --new-tokens 32 --warmup-prompts 1
    "$NSYS" export --type=sqlite --output="$NAME.sqlite" "$NAME.nsys-rep"
done

for RANGE in moe attention; do
    echo "===== $RANGE blocks"
    python analysis/analyse_timeline.py --range "$RANGE" --last 0 results/model_timeline_*_"${SLURM_JOB_ID}".sqlite
done
