# cs-471-project

Characterizing routing imbalance in expert-parallel Mixture-of-Experts inference.

## Baseline benchmark

`moe_bench` is a standalone MoE-layer microbenchmark using the expert shape of
Qwen1.5-MoE-A2.7B: 60 experts, top-4 routing, hidden size 2048, about 17 MB of
bf16 weights per expert. The default is the proposal's **No Load Balance** setup:
contiguous placement with no replicas.

| File | Contents |
|---|---|
| `moe_bench/routing.py` | Zipf routing, contiguous and EPLB placement, replica choice, imbalance metrics |
| `moe_bench/moe_layer.py` | Experts, the per-rank traffic plan, dispatch, compute and combine |
| `moe_bench/benchmark.py` | Skew and batch-size sweep, CUDA event timing, NVTX ranges, CSV output |
| `analysis/analyse_results.py` | Phase tables, Q3 correlations and cost fits from the CSVs |
| `analysis/analyse_timeline.py` | How busy the GPU was during compute, from Nsight Systems traces |
| `jobs/` | Kuma job scripts |
| `tests/` | Logic tests that run on a CPU, with threads standing in for GPUs |

Batch size is the total number of tokens per step across all GPUs. Each output
row in the CSV is one GPU in one step. A step takes as long as its slowest GPU.

**New token counts.** In the first sweep, a step ran about 1 ms slower for each
expert token count that GPU had never computed before, and this looks like a
one-time setup cost rather than a load effect. So by default the benchmark replays
the measured routing once, untimed, before timing it (`--warm-passes 1`). The
`new_sizes` column records any that remain in a timed step. Use `--warm-passes 0`
to keep the cost in the timings.

Not included yet: other placements (Q4), attention and the KV cache. The shared
expert is excluded because it moves no data.

### Load balancing

`--balance` picks the policy for Q2. All policies see the same logical routing.

| Value | Physical experts | What it does |
|---|---|---|
| `none` | 60 | Contiguous placement, no replicas |
| `eplb-1.0` | 60 | EPLB moves experts between GPUs |
| `eplb-1.5` | 90 (92 on 4 GPUs) | EPLB also adds replicas of hot experts, and splits each expert's tokens evenly across its copies |

`eplb_placement` follows the single-node policy in
[DeepSeek's EPLB](https://github.com/deepseek-ai/EPLB) (MIT licence). Three choices
to know about:

- **Load input.** EPLB gets each expert's measured assignment count, not its Zipf
  probability. A token picks an expert at most once, so at alpha=1.5 the hottest
  expert has p=0.43 but receives 23% of assignments. Balancing on p leaves GPU
  imbalance near 1.3, and balancing on measured load brings it to about 1.0.
- **Copies on one GPU.** EPLB doesn't stop two copies of an expert sharing a GPU.
  Each copy has its own weights and runs separately, so it counts as another
  activated expert in the `activated` column.
- **Replica choice.** Tokens go round-robin across an expert's copies from a random
  first copy, without preferring a copy on the token's home GPU.

To check `eplb_placement` against DeepSeek's code, clone their repo and run the test
with it on the path. Without it, this test is skipped:

```bash
git clone https://github.com/deepseek-ai/EPLB.git ../EPLB
```

```bash
PYTHONPATH=../EPLB python -m pytest tests/test_routing.py -k open_source
```

### Run on Kuma

Submit from the repo root. The jobs use the group's venv in
`/work/cs-471/the-girls/venv`.

Check correctness first, using `--verify`:

```bash
sbatch jobs/baseline_smoke.sbatch
```

Then run the Q1 sweep, with three seeds and batches up to 65536:

```bash
sbatch jobs/baseline_sweep.sbatch
```

And the control that varies expert size, with the same operations and routing, to
separate software overhead from the cost of reading weights:

```bash
sbatch jobs/baseline_control.sbatch
```

The same control at batches 1024 to 65536, where tokens matter more:

```bash
sbatch jobs/baseline_control_large.sbatch
```

And a Nsight Systems trace, to see whether compute waits on the CPU or the GPU.
Tracing slows kernel launches, so don't use these runs as timing results:

```bash
sbatch jobs/baseline_timeline.sbatch
```

The large-batch control and the trace print their analysis at the end of the log.

Then the Q2 sweep, comparing the three load balancing policies:

```bash
sbatch jobs/eplb_sweep.sbatch
```

For 4 GPUs, run the check and then the same sweep with one larger batch. Each GPU
must hold the same number of experts, so 1.5x uses 92 physical experts there. These
jobs request 32 cores, which keeps the 2-GPU jobs' 8 cores per GPU:

```bash
sbatch jobs/smoke_4gpu.sbatch
```

```bash
sbatch jobs/eplb_sweep_4gpu.sbatch
```

To look into slow dispatch steps on 4 GPUs, this runs one short configuration on
different GPU pairs, then on 4 GPUs as usual, pinned to NUMA nodes (using
`moe_bench/numa_bind.py`), and with 1 thread per process:

```bash
sbatch jobs/diag_4gpu.sbatch
```

Output goes to `results/`, and the printed log to `<job name>_<job id>.out`.

### Analyse the results

```bash
python analysis/analyse_results.py results/sweep_seed*.csv
```

```bash
python analysis/analyse_timeline.py results/timeline_*.sqlite
```

### Run the tests

These need `torch`, `numpy` and `pytest`, but no GPU:

```bash
python -m pytest tests
```
