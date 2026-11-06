"""Pin this process to the CPUs next to its GPU, then run a module as usual.

Launch with torchrun, CUDA_DEVICE_ORDER=PCI_BUS_ID and numeric CUDA_VISIBLE_DEVICES:
    torchrun --standalone --nproc_per_node=4 -m moe_bench.numa_bind moe_bench.benchmark --alphas 1.5
"""
import os
import subprocess
import sys
from pathlib import Path


def parse_cpu_list(text):
    """Turn the kernel's "0-2,8" into {0, 1, 2, 8}."""
    cpus = set()
    for part in text.strip().split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def sysfs_device(bus_id):
    """nvidia-smi's 00000000:1B:00.0 is 0000:1b:00.0 under /sys."""
    return bus_id.strip()[-12:].lower()


def numa_node_of_gpu(gpu):
    query = ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"]
    bus_ids = subprocess.run(query, check=True, capture_output=True, text=True).stdout.splitlines()
    return int(Path(f"/sys/bus/pci/devices/{sysfs_device(bus_ids[gpu])}/numa_node").read_text())


def main():
    rank = int(os.environ["LOCAL_RANK"])
    gpu = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank])
    node = numa_node_of_gpu(gpu)
    node_cpus = parse_cpu_list(Path(f"/sys/devices/system/node/node{node}/cpulist").read_text())
    cpus = node_cpus & os.sched_getaffinity(0)
    if not cpus:
        sys.exit(f"rank {rank}: this job has no CPUs on NUMA node {node}, next to GPU {gpu}")
    os.sched_setaffinity(0, cpus)
    print(f"rank {rank}: GPU {gpu}, NUMA node {node}, pinned to {len(cpus)} CPUs: {sorted(cpus)}", flush=True)
    # The pinning carries over to the new program, and memory then comes from this node.
    os.execv(sys.executable, [sys.executable, "-u", "-m", *sys.argv[1:]])


if __name__ == "__main__":
    main()
