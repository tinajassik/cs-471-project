#!/bin/bash -l
#SBATCH --job-name=eplb_sweep
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out

# Q2 sweep: No Load Balance against EPLB 1.0x (moves experts) and 1.5x (adds replicas).
# All three run in this job on the same GPUs and see the same logical routing, so
# they can be compared directly. Each seed finishes before the next one starts.
# Submit from the repo root: sbatch jobs/eplb_sweep.sbatch

source jobs/env.sh

nvidia-smi topo -m

BATCHES=1,16,64,256,1024,4096,16384,65536

for SEED in 471 472 473; do
    for BALANCE in none eplb-1.0 eplb-1.5; do
        torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark \
            --seed "$SEED" --balance "$BALANCE" --alphas 1,1.5 --batch-sizes "$BATCHES" \
            --output "results/eplb_${BALANCE}_seed${SEED}_${SLURM_JOB_ID}.csv"
    done
done
