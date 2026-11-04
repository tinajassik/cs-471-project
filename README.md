# cs-471-project

Characterizing routing imbalance in expert-parallel Mixture-of-Experts inference.

## Baseline benchmark

`moe_bench` is a standalone MoE-layer microbenchmark using the expert shape of
Qwen1.5-MoE-A2.7B: 60 experts, top-4 routing, hidden size 2048, about 17 MB of
bf16 weights per expert. This baseline is the proposal's **No Load Balance**
setup: contiguous placement with no replicas.

| File | Contents |
|---|---|
| `moe_bench/routing.py` | Zipf routing, contiguous placement, imbalance metrics |
| `moe_bench/moe_layer.py` | Experts, the per-rank traffic plan, dispatch, compute and combine |
| `moe_bench/benchmark.py` | Skew and batch-size sweep, CUDA event timing, NVTX ranges, CSV output |
| `analysis/analyse_results.py` | Phase tables, Q3 correlations and cost fits from the CSVs |
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

Not included yet: EPLB, other placements, attention and the KV cache. The
shared expert is excluded because it moves no data.

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

Output goes to `results/`, and the printed log to `<job name>_<job id>.out`.

### Analyse the results

```bash
python analysis/analyse_results.py results/sweep_seed*.csv
```

### Run the tests

These need `torch`, `numpy` and `pytest`, but no GPU:

```bash
python -m pytest tests
```
