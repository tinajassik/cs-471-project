#!/bin/bash -l
#SBATCH --job-name=baseline_smoke
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out

# Quick correctness check: compares outputs against a single-GPU reference.
# Don't use these timings: --verify adds work and only a few steps run.
# Submit from the repo root: sbatch jobs/baseline_smoke.sbatch

source jobs/env.sh

nvidia-smi topo -m

torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark \
    --alphas 0,1.5 --batch-sizes 1,2,64,1024 --warmup 1 --steps 3 --verify \
    --output "results/smoke_${SLURM_JOB_ID}.csv"
