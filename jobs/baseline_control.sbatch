#!/bin/bash -l
#SBATCH --job-name=baseline_control
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

# Is the roughly 55 us cost per activated expert software overhead or memory?
# These sizes keep the same operations and routing with 1, 1/4, 1/16 and 1/64 of
# the weight bytes. If the cost stays near 55 us, it doesn't come from the weights.
# Submit from the repo root: sbatch jobs/baseline_control.sbatch

source jobs/env.sh

for EXPERT_SIZE in 1408 352 88 22; do
    torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark \
        --expert-size "$EXPERT_SIZE" --alphas 0,1.5 --batch-sizes 1,4,16,64,256 \
        --output "results/control_size${EXPERT_SIZE}_${SLURM_JOB_ID}.csv"
done
