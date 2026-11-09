#!/bin/bash -l
#SBATCH --job-name=replay_sweep
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

# Q5: replay a model's recorded routing in the expert-parallel benchmark, at that model's
# expert shape, under No Load Balance, EPLB 1.0x and EPLB 1.5x. Uses the least, median and
# most skewed layers, and runs uniform routing at the same shape as the control.
# Submit from the repo root, with a trace from jobs/model_trace.sbatch:
#     sbatch --export=ALL,TRACE=results/trace_qwen3-30b-a3b_<job>.npz jobs/replay_sweep.sbatch
# On 4 GPUs (one CPU thread per process, which removes the dispatch stalls):
#     sbatch --gpus-per-node=4 --cpus-per-task=32 --export=ALL,TRACE=... jobs/replay_sweep.sbatch

source jobs/env.sh
nvidia-smi topo -m

: "${TRACE:?set TRACE to a trace .npz from model_trace.sbatch}"
GPUS=${SLURM_GPUS_ON_NODE:-2}
if [ "$GPUS" -ge 4 ]; then export OMP_NUM_THREADS=1; fi
TAG=$(basename "$TRACE" .npz)
LAYERS=$(python analysis/analyse_trace.py "$TRACE" --pick-layers)
echo "replaying layers $LAYERS of $TRACE on $GPUS GPUs"

# Decode batches hold the next token of that many prompts, so they stop at the prompt count.
PREFILL_BATCHES=16,64,256,1024,4096,16384,65536
DECODE_BATCHES=1,4,16,64,256

run() {
    torchrun --standalone --nproc_per_node="$GPUS" -m moe_bench.benchmark "$@"
}

for BALANCE in none eplb-1.0 eplb-1.5; do
    run --trace "$TRACE" --trace-phase prefill --trace-layers "$LAYERS" --balance "$BALANCE" \
        --batch-sizes "$PREFILL_BATCHES" --output "results/replay_${TAG}_prefill_${BALANCE}_${GPUS}gpu_${SLURM_JOB_ID}.csv"
    run --trace "$TRACE" --trace-phase decode --trace-layers "$LAYERS" --balance "$BALANCE" \
        --batch-sizes "$DECODE_BATCHES" --output "results/replay_${TAG}_decode_${BALANCE}_${GPUS}gpu_${SLURM_JOB_ID}.csv"
done
run --shape "$TRACE" --alphas 0 --batch-sizes 1,4,$PREFILL_BATCHES \
    --output "results/replay_${TAG}_uniform_${GPUS}gpu_${SLURM_JOB_ID}.csv"

python analysis/analyse_replay.py results/replay_"${TAG}"_*_"${GPUS}"gpu_"${SLURM_JOB_ID}".csv
