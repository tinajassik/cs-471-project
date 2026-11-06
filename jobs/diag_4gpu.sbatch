#!/bin/bash -l
#SBATCH --job-name=diag_4gpu
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

# Why do 4-GPU runs have slow dispatch steps at batch 1024 and up, and take far longer
# outside the timed steps? All runs share one node, so they compare directly:
#   2 GPUs on 0,1 and on 2,3: is it those GPUs, or having 4 processes?
#   2 GPUs on 0,2: is it crossing between NUMA nodes?
#   4 GPUs, as in the sweep, then pinned to their NUMA nodes, then with 1 thread each.
# Log lines carry the time, so gaps between them show each configuration's wall time.
# Submit from the repo root: sbatch jobs/diag_4gpu.sbatch

source jobs/env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID

nvidia-smi topo -m
lscpu | grep -E "^(CPU\(s\)|Thread|Core|Socket|NUMA)"
taskset -cp $$

stamp() {
    while IFS= read -r line; do
        echo "$(date +%T) $line"
    done
}

run() {
    local name=$1 gpus=$2 count=$3 module=$4
    echo "$(date +%T) ===== $name: GPUs $gpus, $OMP_NUM_THREADS threads per process"
    # module is left unquoted, because the pinned run gives two words.
    CUDA_VISIBLE_DEVICES=$gpus torchrun --standalone --nproc_per_node="$count" -m $module \
        --seed 471 --balance none --alphas 1.5 --batch-sizes 1024,4096,16384,65536 \
        --output "results/diag_${name}_${SLURM_JOB_ID}.csv" 2>&1 | stamp
}

# Every 30 s, record what each process is doing: CPU, thread count and CPU use so far.
while true; do
    date +%T
    ps -u "$USER" -o pid,psr,pcpu,nlwp,comm --sort=-pcpu | head -7
    sleep 30
done > "results/diag_cpu_${SLURM_JOB_ID}.log" &
MONITOR=$!

run gpus01 0,1 2 moe_bench.benchmark
run gpus23 2,3 2 moe_bench.benchmark
run gpus02 0,2 2 moe_bench.benchmark
run all4 0,1,2,3 4 moe_bench.benchmark
run all4_pinned 0,1,2,3 4 "moe_bench.numa_bind moe_bench.benchmark"
OMP_NUM_THREADS=1 run all4_1thread 0,1,2,3 4 moe_bench.benchmark

kill "$MONITOR"
echo "$(date +%T) DIAGNOSTIC FINISHED"
