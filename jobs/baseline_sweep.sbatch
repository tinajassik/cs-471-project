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

# First Q1 sweep of skew against batch size, on the No Load Balance baseline.
# Submit from the repo root: sbatch jobs/baseline_sweep.sbatch

source jobs/env.sh

nvidia-smi topo -m

# One layer: a repeated expert's weights may still be in the GPU cache.
torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark \
    --layers 1 --output "results/baseline_1layer_${SLURM_JOB_ID}.csv"

# 24 layers, one per step, like the real model: fewer weights should be cached.
# Comparing the two runs shows whether caching hides the memory-bound regime.
torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark \
    --layers 24 --output "results/baseline_24layers_${SLURM_JOB_ID}.csv"
