#!/bin/bash -l
#SBATCH --job-name=topk_sweep
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out

# How do skew and EPLB change with top-k? Repeats the Q1-Q2 sweep with Zipf routing at
# other top-k values (k=4 is the existing sweep), and at Qwen3-30B-A3B's shape (128 experts, top-8).
# Submit from the repo root: sbatch jobs/topk_sweep.sbatch
# Fewer seeds or values: sbatch --export=ALL,SEEDS=471,TOPK="2 8" jobs/topk_sweep.sbatch

source jobs/env.sh
nvidia-smi topo -m

SEEDS=${SEEDS:-"471 472 473"}
TOPK=${TOPK:-"2 8"}
BATCHES=1,16,64,256,1024,4096,16384,65536

run() {
    torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark --batch-sizes "$BATCHES" "$@"
}

for SEED in $SEEDS; do
    for CONFIG in $(for K in $TOPK; do echo "qwen1.5-moe:$K"; done) "qwen3-30b-a3b:8"; do
        SHAPE=${CONFIG%:*}
        K=${CONFIG#*:}
        NAME="results/topk_${SHAPE}_k${K}_seed${SEED}"
        run --seed "$SEED" --shape "$SHAPE" --top-k "$K" --balance none --alphas 0,1,1.5 \
            --output "${NAME}_none_${SLURM_JOB_ID}.csv"
        for BALANCE in eplb-1.0 eplb-1.5; do
            run --seed "$SEED" --shape "$SHAPE" --top-k "$K" --balance "$BALANCE" --alphas 1,1.5 \
                --output "${NAME}_${BALANCE}_${SLURM_JOB_ID}.csv"
        done
    done
done
