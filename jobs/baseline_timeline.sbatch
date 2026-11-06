#!/bin/bash -l
#SBATCH --job-name=baseline_timeline
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

# Is the compute phase waiting on the CPU or on the GPU? Traces every kernel with
# Nsight Systems and prints how busy the GPU was. Tracing slows kernel launches, so
# use these runs for that question only, not as timing results.
# Submit from the repo root: sbatch jobs/baseline_timeline.sbatch

module purge
module load gcc cuda
module load nvhpc
NSYS=$(command -v nsys)

source jobs/env.sh

mkdir -p results

# Small batches at full and tiny expert size, then larger batches at full size.
for CONFIG in "64 1408" "64 22" "4096 1408" "65536 1408"; do
    read -r BATCH EXPERT_SIZE <<< "$CONFIG"
    NAME="results/timeline_batch${BATCH}_size${EXPERT_SIZE}_${SLURM_JOB_ID}"
    # analyse_timeline.py reads the last 20 compute phases, which are these timed steps.
    "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --force-overwrite=true -o "$NAME" \
        torchrun --standalone --nproc_per_node=2 -m moe_bench.benchmark \
        --alphas 0 --batch-sizes "$BATCH" --expert-size "$EXPERT_SIZE" --warmup 5 --steps 20 \
        --output "$NAME.csv"
    "$NSYS" export --type=sqlite --output="$NAME.sqlite" "$NAME.nsys-rep"
done

python analysis/analyse_timeline.py results/timeline_*_"${SLURM_JOB_ID}".sqlite
