# Provenance of the committed results

Every result file under `results/` comes from one run of `scripts/run_gpu_plan.py`
on 2026-11-20 (local time UTC+05:30), started at commit `c4c111b` with the package
source unchanged. Each file records where it came from, so the claims below can be
checked against the files themselves.

## Code and environment

| Item | Value |
|---|---|
| Commit | `c4c111b` (Measure cuDNN attention in the bf16 inference sweep) |
| Package source hash | `c4adb70bef570702...` in every result file (`source_sha256`) |
| `git_dirty` | `false` in every result file: `src/ssmbench` matched the commit |
| Environment | `results/ENVIRONMENT.json`: RTX 4050 Laptop GPU (6140.5 MiB), driver 610.74, torch 2.14.1+cu130, CUDA 13.0, Python 3.13.7, Windows 11, the full package list and the `uv.lock` hash |

`git_dirty` covers the package source only. During the run the working tree also held
an uncommitted change to `scripts/make_tables.py` (the environment table, committed
afterwards as `4172acb`); the tables were regenerated from the committed script, and
`python scripts/make_tables.py --check` passes on the committed results.

## Jobs

The plan ran these jobs in order, each logging to `results/logs/<job>.log`:

| Job | Started (local) | Output |
|---|---|---|
| `environment` | 02:33, 02:38, 02:42 (the last one is kept) | `results/ENVIRONMENT.json` |
| `smoke` | 02:33, 02:38 | `results/logs/smoke.json` (bitwise-repeatable training steps for every variant; every pinned attention kernel once) |
| `train-parity` | 02:42 | `results/train/parity/<variant>/seed{0,1,2}.json` (7 variants) |
| `train-selective_copy` | 03:36 | `results/train/selective_copy/...` (6 variants) |
| `train-exact_copy` | 04:27 | `results/train/exact_copy/...` (6 variants) |
| `train-exact_copy_varlen` | 06:45 | `results/train/exact_copy_varlen/...` (6 variants) |
| `train-imdb` | 08:50 | `results/train/imdb/...` (5 variants; Mamba-1 is not in the config) |
| `summarize` | 10:43 | `results/train/<task>/summary.json` |
| `efficiency-fp32`, `efficiency-bf16` | 10:44, 11:03 | `results/efficiency/sweep_{fp32,bf16}.json` |
| `inference-fp32`, `inference-bf16` | 11:13, 11:18 | `results/inference/decode_{fp32,bf16}.json` |
| `numerics` | 11:21 | `results/numerics/numerics.json` |
| `compile` | 11:22 | `results/compile/compile.json` |
| `figures`, `tables` | 11:24 | `figures/`, `reports/figures/`, `reports/tables/`, `results/SUMMARY.md` |

The first smoke attempt (02:33) failed while loading IMDb under the dataset id
`imdb`, which the installed `huggingface_hub` rejects; the loader was switched to
`stanfordnlp/imdb` at a pinned revision (`5587386`) and the second attempt passed.
Both attempts are in `results/logs/smoke.log`.

## Training runs

Each `seed<k>.json` holds the config and its SHA-256, the run seed, the parameter
count, the per-epoch history on the validation set, the final test evaluations, the
attention kernel detected for the Transformers, wall time, peak memory, a spill flag
and the provenance above. All 90 runs (30 variant and task pairs, seeds 0, 1, 2) ran
with deterministic algorithms; none resumed from a checkpoint
(`resumed_from_epoch: null`) and none exceeded free GPU memory
(`gpu_memory_spilled: false`).

## Benchmarks

The sweep files keep their config, one entry per (model, length) or (model, seed)
point with its provenance, the GPU clock and power state when the point started, and
for the timing sweeps the peak allocated and reserved memory and a spill flag. A
spilled point ran past free GPU memory and was paged to system memory by the Windows
display driver, so its time is not a GPU-resident measurement; the tables mark it.
FlashAttention points are recorded as unsupported because the Windows torch wheel
does not include that kernel. Inductor was not run because Triton is not installed.

## Logs

The job logs are committed as written, except that the local repository path and the
user's home directory were replaced by `.` and `~` before committing. They include
the kineto profiler messages printed while detecting the attention kernel.

## Earlier results

The single-seed result files and figures were removed in `ab29379` and `fa2c2d4`;
they described models with defects fixed before this run. They remain in the git
history, together with the detailed provenance notes written for them:
`git show c4c111b:results/PROVENANCE.md`.
