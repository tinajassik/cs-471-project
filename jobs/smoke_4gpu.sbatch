#!/bin/bash -l
#SBATCH --job-name=smoke_4gpu
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

# Quick correctness check of every balancing policy on 4 GPUs: compares outputs
# against a single-GPU reference, which doesn't depend on placement or replicas.
# Don't use these timings: --verify adds work and only a few steps run.
# Submit from the repo root: sbatch jobs/smoke_4gpu.sbatch

source jobs/env.sh

nvidia-smi topo -m

for BALANCE in none eplb-1.0 eplb-1.5; do
    torchrun --standalone --nproc_per_node=4 -m moe_bench.benchmark \
        --balance "$BALANCE" --alphas 0,1.5 --batch-sizes 1,2,64,1024 --warmup 1 --steps 3 --verify \
        --output "results/smoke_4gpu_${BALANCE}_${SLURM_JOB_ID}.csv"
done
