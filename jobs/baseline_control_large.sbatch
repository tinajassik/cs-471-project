#!/bin/bash -l
#SBATCH --job-name=baseline_control_large
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

# Does the per-token cost depend on the weights? baseline_control.sbatch stops at
# batch 256, where tokens barely matter. These batches have about 30 activated
# experts per GPU throughout, so the slope against tokens is the per-token cost.
# Submit from the repo root: sbatch jobs/baseline_control_large.sbatch

source jobs/env.sh

for EXPERT_SIZE in 1408 352 88 22; do
    torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark \
        --expert-size "$EXPERT_SIZE" --alphas 0,1.5 --batch-sizes 1024,4096,16384,65536 \
        --output "results/control_large_size${EXPERT_SIZE}_${SLURM_JOB_ID}.csv"
done

python analysis/analyse_results.py results/control_large_size*_"${SLURM_JOB_ID}".csv
