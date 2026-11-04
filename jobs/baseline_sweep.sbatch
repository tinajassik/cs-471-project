#!/bin/bash -l
#SBATCH --job-name=baseline_sweep
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

# Q1 sweep of skew against batch size, on the No Load Balance baseline.
# Submit from the repo root: sbatch jobs/baseline_sweep.sbatch

source jobs/env.sh

nvidia-smi topo -m

BATCHES=1,4,16,64,256,1024,4096,16384,65536

# Several seeds, because which experts are hot depends on the seed.
for SEED in 471 472 473; do
    torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark \
        --seed "$SEED" --batch-sizes "$BATCHES" \
        --output "results/sweep_seed${SEED}_${SLURM_JOB_ID}.csv"
done

# 24 layers, one per step, like the real model: checks that caching stays invisible.
torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark \
    --seed 471 --layers 24 --batch-sizes "$BATCHES" \
    --output "results/sweep_24layers_seed471_${SLURM_JOB_ID}.csv"

# No warm pass, for the same seed: if new token counts cause the slow steps, they
# should appear here and be gone from the seed 471 run above.
torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark \
    --seed 471 --warm-passes 0 --batch-sizes 1024,4096 \
    --output "results/sweep_nowarm_seed471_${SLURM_JOB_ID}.csv"
