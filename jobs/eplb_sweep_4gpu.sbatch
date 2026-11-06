#!/bin/bash -l
#SBATCH --job-name=eplb_sweep_4gpu
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

# eplb_sweep.sbatch on 4 GPUs. It keeps 8 cores per GPU, like the 2-GPU jobs, because
# launches depend on the CPU. The same batches compare directly with the 2-GPU sweep,
# and batch 131072 matches the tokens per GPU of batch 65536 there.
# Submit from the repo root: sbatch jobs/eplb_sweep_4gpu.sbatch

source jobs/env.sh

nvidia-smi topo -m

BATCHES=1,16,64,256,1024,4096,16384,65536,131072

for SEED in 471 472 473; do
    for BALANCE in none eplb-1.0 eplb-1.5; do
        torchrun --standalone --nproc_per_node=4 -m moe_bench.benchmark \
            --seed "$SEED" --balance "$BALANCE" --alphas 1,1.5 --batch-sizes "$BATCHES" \
            --output "results/eplb_4gpu_${BALANCE}_seed${SEED}_${SLURM_JOB_ID}.csv"
    done
done
