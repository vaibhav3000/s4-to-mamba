# Provenance of committed results

This file records where every committed result and figure came from, which code produced it,
and which later fixes make it stale. It covers every tracked file under `results/`,
`figures/` and `reports/figures/`.

## Status

Every number in `results/*.json`, `figures/*`, `reports/figures/*` and every number quoted in
the README, the report and the docs was produced between 2026-09-16 and 2026-09-18 by code
that predates the fixes listed in section 1. These files are kept exactly as committed, as a
record of what was measured, and should be read as stale. No committed number has been edited.

- The S4D, Mamba-1, Mamba-2 and Mamba-3 task results come from model or training code that has
  since been corrected (section 1).
- The Transformer task metrics are not changed by any of these fixes. They still come from the
  same pre-fix runs, carry no record of the code version, and will be regenerated with the rest.
- `results/efficiency_gpu.json` was stitched together from eight benchmark processes that ran
  at least three different versions of the benchmark code (section 7.2). Every one of its
  training-step numbers was measured without an LM head.

Conventions. Commit hashes are short hashes on branch `upgrade/nov-2026`. Source line numbers
are given at the commit named next to them, written `file:line @ commit`. `log:N` is line N of
`experiments.log` (834 lines; the home-path redaction in 8a1ed04 changed no line numbers). If
the log is edited later, find the line by its content. "Verified" means checked against a
committed file or git object. "Inferred" means the evidence fits but does not prove it.

## 1. Fixes that make the results stale

| ID | What was wrong in the code that produced the results | Where (historical) | Fixed in | What it affects |
|---|---|---|---|---|
| P1 | Mamba-2 and Mamba-3 expanded grouped B and C with `repeat_interleave` on the state axis. With `n_groups=1` (every config), head h read N/H components of B and C, each repeated H times, instead of the group's full N-vector | `mamba2_minimal.py:119-120`, `mamba3_minimal.py:170-171` @ b327a35, bcc3f7d, 246f4cb | 0a2461a | every mamba2 and mamba3 result, task and efficiency |
| P2 | S4D ZOH input matrix was delta\*B\*(e^z-1) instead of delta\*B\*(e^z-1)/z: `z.where(small, ones)` evaluates to `where(small, z, 1)`, so the divisor was 1 | `s4_minimal.py:53` @ b327a35 (unchanged until the fix) | 1751001 (4a8a6f8 then evaluates the factor without float32 cancellation) | every s4d result, task and efficiency |
| D10 | Weight decay went to every parameter with more than one dimension, including 2-D SSM dynamics parameters. The fix exempts S4D `log_A_real`, `A_imag`, `dt_param`, `B_re`, `B_im`, Mamba-1 `A_log`, Mamba-3 `a_real_log`, `theta` | `trainer.py:159-161` @ b327a35, `trainer.py:172-174` @ 246f4cb | dcb4dd1 | task training of s4d, mamba and mamba3 only. Not mamba2 (its `A_log` and `D` are 1-D and were already exempt), not the Transformer, not the efficiency benchmark (it builds its own `AdamW` over all backbone parameters, `efficiency.py:114` @ bcc3f7d) |
| D17 | Mamba-2's chunked scan took `log(clamp_min(exp(delta*A), 1e-12))` instead of using delta\*A, which loses the gradient once the exponent underflows | `mamba2_minimal.py:117`, `:165` @ b327a35, bcc3f7d | 52f53bf (c4e53e7 then accumulates every in-chunk decay sum directly) | every mamba2 result; every mamba2 run used the chunked scan (section 6) |
| P16 | The efficiency train step had no LM head: backbone features of width d_model were fed to cross-entropy as d_model-way logits | `efficiency.py:115-122` @ b327a35, `:117-124` @ bcc3f7d | 0b58559 | `train_step_ms`, `train_step_ms_std`, `train_tokens_per_s`, `peak_train_mb` of every efficiency entry, and the `oom_train`/`oom` flags, which came from the same step |
| P6 | The trainer read `torch.cuda.max_memory_allocated()` without ever resetting it, while one process trained every model of a config | `trainer.py:224` @ b327a35, `:237` @ 246f4cb; `scripts/run_training.py:38-48` @ b327a35 | 8d15605, 9824d80 | `peak_gpu_mb` of training files that were not the first model in their process (section 5) |
| P17 | S4D `step()` rebuilt its parameters and ZOH discretization on every token; the benchmark never called `eval()` before inference and decode timing | `s4_minimal.py:142-143` @ b327a35 | bab71db | s4d `decode_ms` and `decode_per_token_ms`. `eval()` changes no number: every backbone has dropout 0.0 by default and no train-mode branch (`s4_minimal.py:153,191`, `transformer_minimal.py:48,71` @ bcc3f7d) |
| P3 | Decode fields mix definitions. The Transformer entries carry the first label and forward time divided by T; mamba2 and mamba3 "decode" is one no-grad forward over T tokens divided by T, labeled as a sequential scan | `efficiency.py:167-181` @ b327a35; `:169-184` @ bcc3f7d | bd7d6f1 (Mamba-2/3 now report `prefill_per_token_ms`) | decode fields of every transformer, mamba2 and mamba3 entry |
| P4 | Only allocated memory was recorded. Five entries exceed the 6140.5 MiB device total, which is only possible through the Windows WDDM system-memory fallback | `efficiency.py:130`, `:151` @ b327a35; `:132`, `:153` @ bcc3f7d | 8d15605, 9824d80 (peak reserved memory, device total, memory available at the start and a spill flag are now recorded) | five entries (section 7.4) |
| P5, D11 | Resume merged entries across code versions and rewrote the top-level `environment` on every restart; entries carry no code version; the handlers labeled any `torch.AcceleratorError` as out of memory | `efficiency.py:192-243` @ bcc3f7d | 6999011 (D11), 7d8c342 and 7c7dc13 (P5) | the whole efficiency file (section 7.2) |

No fix changes a parameter shape: `git diff 8d70fbd 54298dd -- src/ssmbench/models` changes
one `nn.Parameter` line, a rename in the S4D initializer with bitwise-identical values, and adds
or removes no `nn.Parameter`, `nn.Linear`, `nn.Conv1d` or `nn.Embedding`. The `n_parameters`
fields are therefore not stale through any mechanism above. D10 changes optimizer groups, not
parameters. The fixed benchmark still counts backbone parameters only in `n_parameters` and
adds `n_parameters_with_embed_and_head` for the new LM head.

Other changes on this branch that do not affect any committed number: 14d37f0 (scans run in
float64 for float64 inputs; float32 outputs bitwise unchanged), b910a5d (scans ignore bf16
autocast; no committed config enables `amp`), and 4a8a6f8, which fixes float32 cancellation that
existed only between 1751001 and 4a8a6f8 (the pre-fix divisor of 1 did not cancel).

## 2. Commits that touched these files

| Commit | Date (+0530) | Subject |
|---|---|---|
| b327a35 | 2026-09-16 21:38:51 | s4-to-mamba: minimal S4D/Mamba/Mamba-2-SSD/Mamba-3 implementations, equivalence tests, parity+selective-copy+IMDb results, docs, figures |
| bcc3f7d | 2026-09-16 23:38:20 | s4-to-mamba: complete 1K-32K efficiency sweep, mamba2 IMDb result, figures, resume support in benchmark harness |
| 1002451 | 2026-09-17 02:56:20 | README: embed evolution diagram and benchmark figures; OOM-annotated figures regenerated |
| 246f4cb | 2026-09-18 03:38:32 | literature reproduction: Repeat After Me exact-copy task, 4-model grid, task-formulation study, docs and figures |
| 1c30806 | 2026-09-19 02:09:39 | docs: add IEEE-style technical report (PDF + LaTeX source) |

The result JSON files are byte-identical between 8d70fbd (the last commit on `main`) and every
later commit on this branch: `git diff --stat 8d70fbd HEAD -- 'results/*.json'` prints nothing.
The figures were redrawn from them in 54298dd and `experiments.log` was redacted in 8a1ed04;
the tables below describe the history up to 8d70fbd.

A note on `git log --follow`: for 11 of these files it also lists commits in which the file did
not exist. Git's copy detection pairs a new PDF with an older, similar PDF (for example, in
bcc3f7d it reports `figures/fig_efficiency_decode.pdf` as a 63% copy of
`figures/fig_selective_copy.pdf`) and `results/imdb_mamba2.json` with `results/imdb_s4d.json`.
The tables below give the plain `git log -- <file>` history and mark the extra commits that
`--follow` adds.

### 2.1 `results/`

| File | Added in | Changed since | Extra commits under `--follow` (file did not exist) |
|---|---|---|---|
| `results/parity_{transformer,s4d,mamba,mamba2,mamba3}.json` (5 files) | b327a35 | never | none |
| `results/selective_copy_{transformer,s4d,mamba,mamba2,mamba3}.json` (5 files) | b327a35 | never | none |
| `results/imdb_transformer.json`, `results/imdb_s4d.json` | b327a35 | never | none |
| `results/imdb_mamba2.json` | bcc3f7d | never | b327a35 |
| `results/efficiency_gpu.json` | bcc3f7d | never | none |
| `results/exact_copy_{transformer,s4d,mamba,mamba2}.json` (4 files) | 246f4cb | never | none |

### 2.2 `figures/` and `reports/figures/`

Every PDF records its `/CreationDate`, and all were written by "Matplotlib pdf backend v3.10.8".
Where a commit changed a PDF without changing its size, the only bytes that differ are inside
the `/CreationDate` string (verified for every such pair by comparing the git blobs), so those
commits re-saved the same figure.

| File | Commits (plain `git log`, oldest first) | What changed | Extra commits under `--follow` |
|---|---|---|---|
| `figures/fig_parity.png` | b327a35 | added | none |
| `figures/fig_parity.pdf` | b327a35, bcc3f7d, 1002451, 246f4cb | added in b327a35; the other three only re-saved it (CreationDate) | none |
| `figures/fig_selective_copy.png` | b327a35 | added | none |
| `figures/fig_selective_copy.pdf` | b327a35, bcc3f7d, 1002451, 246f4cb | added in b327a35; the other three only re-saved it | none |
| `figures/fig_imdb_quality.png` | b327a35, bcc3f7d | b327a35 shows Transformer and S4D; bcc3f7d adds Mamba-2 | none |
| `figures/fig_imdb_quality.pdf` | b327a35, bcc3f7d, 1002451, 246f4cb | content changed in bcc3f7d (Mamba-2 added); 1002451 and 246f4cb only re-saved it | none |
| `figures/fig_learning_curves.png` | b327a35, bcc3f7d | content changed in bcc3f7d | none |
| `figures/fig_learning_curves.pdf` | b327a35, bcc3f7d, 1002451, 246f4cb | content changed in bcc3f7d; 1002451 and 246f4cb only re-saved it | none |
| `figures/fig_efficiency_{latency,memory,throughput,decode}.png` (4 files) | bcc3f7d, 1002451 | added in bcc3f7d; 1002451 added the OOM markers (`scripts/make_figures.py` changed in the same commit) | none |
| `figures/fig_efficiency_{latency,memory,throughput,decode}.pdf` (4 files) | bcc3f7d, 1002451, 246f4cb | as the PNGs; 246f4cb only re-saved them | b327a35 |
| `figures/fig_exact_copy.png` | 246f4cb | added | none |
| `figures/fig_exact_copy.pdf` | 246f4cb | added | b327a35, bcc3f7d, 1002451 |
| `figures/fig_task_formulation.png` | 246f4cb | added | none |
| `figures/fig_task_formulation.pdf` | 246f4cb | added | b327a35, bcc3f7d, 1002451 |
| `reports/figures/{parity,selcopy,imdb,efficiency}.pdf` (4 files) | 1c30806 | added; CreationDate 2026-09-18 22:09:22 to 22:09:23 | b327a35, bcc3f7d, 1002451, 246f4cb |

No committed script writes `reports/figures/*.pdf`: `git grep savefig` finds only
`scripts/make_figures.py`, which writes `figures/fig_*.{png,pdf}`. Their content, read from the
axis text: `parity.pdf` is final token accuracy for the five parity models with a chance line,
`selcopy.pdf` is final token accuracy and sequence success for the five selective-copy models,
`imdb.pdf` is accuracy and AUC for the three IMDb models, and `efficiency.pdf` has two panels,
train step time and peak GPU memory against sequence length for all five models.

## 3. Training results: inventory

All 17 training files have the same `environment` block:
`{"cuda": "13.0", "device": "cuda", "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU", "torch": "2.14.0+cu130"}`.
It records no Python version, driver, OS, git commit or timestamp. Every file's `model`,
`train_config` and `data_config` snapshot equals the current `configs/*.yaml` entry (checked by
loading both; the configs have not changed since except for comments and the removal of the
`mamba` and `mamba3` lines from `configs/imdb.yaml`).

| File | `name` | `model.name` | `train_config.seed` | `wall_time_seconds` | `n_parameters` | `peak_gpu_mb` | epochs |
|---|---|---|---|---|---|---|---|
| `results/parity_transformer.json` | `parity_transformer` | transformer | 42 | 13.7 | 100354 | 109.5 | 20 |
| `results/parity_s4d.json` | `parity_s4d` | s4d | 42 | 17.8 | 33794 | 109.5 | 20 |
| `results/parity_mamba.json` | `parity_mamba` | mamba | 42 | 174.5 | 78210 | 898.0 | 20 |
| `results/parity_mamba2.json` | `parity_mamba2` | mamba2 | 42 | 20.0 | 68786 | 898.0 | 20 |
| `results/parity_mamba3.json` | `parity_mamba3` | mamba3 | 42 | 402.3 | 69562 | 1701.3 | 20 |
| `results/selective_copy_transformer.json` | `selective_copy_transformer` | transformer | 42 | 44.7 | 401940 | 131.7 | 12 |
| `results/selective_copy_s4d.json` | `selective_copy_s4d` | s4d | 42 | 54.5 | 64020 | 131.7 | 12 |
| `results/selective_copy_mamba.json` | `selective_copy_mamba` | mamba | 42 | 307.3 | 238868 | 513.2 | 12 |
| `results/selective_copy_mamba2.json` | `selective_copy_mamba2` | mamba2 | 42 | 60.3 | 242532 | 513.2 | 12 |
| `results/selective_copy_mamba3.json` | `selective_copy_mamba3` | mamba3 | 42 | 621.8 | 243564 | 1682.1 | 12 |
| `results/imdb_transformer.json` | `imdb_transformer` | transformer | 42 | 85.8 | 3353602 | 1211.1 | 3 |
| `results/imdb_s4d.json` | `imdb_s4d` | s4d | 42 | 57.5 | 2677762 | 1211.1 | 3 |
| `results/imdb_mamba2.json` | `imdb_mamba2` | mamba2 | 42 | 189.4 | 3034786 | 2865.0 | 3 |
| `results/exact_copy_transformer.json` | `exact_copy_transformer` | transformer | 42 | 34.4 | 602782 | 518.5 | 6 |
| `results/exact_copy_s4d.json` | `exact_copy_s4d` | s4d | 42 | 31.2 | 95902 | 518.5 | 6 |
| `results/exact_copy_mamba2.json` | `exact_copy_mamba2` | mamba2 | 42 | 79.2 | 363670 | 1246.3 | 6 |
| `results/exact_copy_mamba.json` | `exact_copy_mamba` | mamba | 42 | 1000.0 | 358174 | 3050.4 | 6 |

## 4. Which code produced each result family

| Family | Files | When | Code | Evidence |
|---|---|---|---|---|
| Parity | `parity_*.json` (5) | 2026-09-16, before 21:37:55 | b327a35, the root commit. The working tree at run time was not recorded, so "b327a35" means "consistent with b327a35" | The torch UserWarning at log:2 cites `trainer.py:207`, which is `ep_loss += float(loss)` @ b327a35 (and @ bcc3f7d, whose trainer is identical); by 246f4cb that statement had moved to line 220. `figures/fig_parity.pdf` @ b327a35 has CreationDate 2026-09-16 21:37:55, before the commit at 21:38:51 |
| Selective copy | `selective_copy_*.json` (5) | same | b327a35 | warning at log:110 cites `trainer.py:207` |
| IMDb, Transformer and S4D | `imdb_transformer.json`, `imdb_s4d.json` | same | b327a35 | warning at log:181 cites `trainer.py:207`; `fig_imdb_quality.pdf` @ b327a35 (CreationDate 21:37:55) shows only these two models |
| IMDb, Mamba-2 | `imdb_mamba2.json` | started around the b327a35 commit, finished before bcc3f7d | b327a35 trainer and b327a35 `mamba2_minimal.py` (the pre-grouping chunked scan) | b327a35's copy of `experiments.log` has 199 lines and ends with the `imdb_mamba2` header and warning (log:196-199); the epoch lines (log:200-202) and the JSON first appear in bcc3f7d. The warning cites `trainer.py:207`. Later that evening efficiency run 2 still executed b327a35's `mamba2_minimal.py` (section 7.2), so this run did too (inferred). The chunk grouping that bcc3f7d added is documented as giving identical outputs and does not touch P1 or D17 |
| Efficiency | `efficiency_gpu.json` | 2026-09-16, eight processes, the earliest timestamp log:209 (21:41:16), committed 23:38:20 | run 1 matches b327a35 exactly; runs 2 to 7 executed uncommitted intermediate versions; run 8 fits bcc3f7d | section 7.2 |
| Exact copy | `exact_copy_*.json` (4) | before 2026-09-18 03:33:37 | 246f4cb trainer and data code; model code as in bcc3f7d (246f4cb changed no file under `src/ssmbench/models`) | warning at log:805 cites `trainer.py:220`, which is `ep_loss += float(loss)` @ 246f4cb; `fig_exact_copy.pdf` CreationDate 2026-09-18 03:33:37; files added in 246f4cb |

How the training processes split, from the `=== running ... ===` headers and the torch
UserWarning. Python prints a given warning once per source location per process, so each
reappearance of the `trainer.py` warning marks a new process.

| Process | Log lines | Models, in order | Result files |
|---|---|---|---|
| parity | 1-108 | transformer, s4d, mamba, mamba2, mamba3 | 5 |
| selective copy | 109-176 | transformer, s4d, mamba, mamba2, mamba3 | 5 |
| IMDb 1 | 177-191 | transformer, s4d, then `imdb_mamba` started (log:191) and produced no epoch line | 2 |
| IMDb 2 | 192-195 | `imdb_mamba` again (header log:192, fresh warning log:193), no epoch line | none |
| IMDb 3 | 196-202 | `imdb_mamba2` only (header log:196, fresh warning log:197, epochs log:200-202, no later header) | 1 |
| exact copy | 804-834 | transformer, s4d, mamba2, mamba | 4 |

Evidence that `imdb_mamba2` was trained in a separate process (verified unless marked): the
warning reappears at log:197 right after its header; process IMDb 2 started with
`imdb_mamba`, not `imdb_transformer`, so it was launched on a model subset (the script has a
`--models` option, `scripts/run_training.py:27` @ b327a35); IMDb 3 prints no header after
`imdb_mamba2`, so it ran that model alone (inferred: `--models mamba2`); the JSON was committed
two hours after the other IMDb files (bcc3f7d, not b327a35); and at b327a35 the README's
IMDb table row for mamba2 only points to `results/imdb_mamba2.json` (README.md:88 @ b327a35),
while the figure at b327a35 has no Mamba-2 bar. Whether IMDb 3 overlapped on the GPU with efficiency run 1 cannot be
told from the log; its `wall_time_seconds` is 189.4 and run 1 logged its first timestamp at
21:41:16, under three minutes after the b327a35 commit.

At b327a35 `configs/imdb.yaml` still listed all five models; e7efd2e (2026-10-03) commented
out `mamba` and `mamba3` with the note "intentionally NOT trained on IMDb" and ba7eeae removed
the commented lines. `imdb_mamba.json` and `imdb_mamba3.json` never existed in any commit.

## 5. Training results: staleness per file and field

Mechanisms that change task metrics (`history[*]` values, `final_by_length`), `epoch_seconds`
and `wall_time_seconds`:

| Model | Files | P1 | P2 | D10 | D17 | Task metrics |
|---|---|---|---|---|---|---|
| transformer | `{parity,selective_copy,imdb,exact_copy}_transformer.json` (4) | no | no | no | no | not changed by any fix; same pre-fix runs, no code version recorded |
| s4d | `{parity,selective_copy,imdb,exact_copy}_s4d.json` (4) | no | yes | yes | no | stale |
| mamba | `{parity,selective_copy,exact_copy}_mamba.json` (3) | no | no | yes | no | stale |
| mamba2 | `{parity,selective_copy,imdb,exact_copy}_mamba2.json` (4) | yes | no | no | yes | stale |
| mamba3 | `{parity,selective_copy}_mamba3.json` (2) | yes | no | yes | no | stale |

Other fields:

- `n_parameters`: no mechanism changes it (section 1).
- `train_config`, `data_config`, `model`: match the current configs. D10 changes what
  `weight_decay` applies to for s4d, mamba and mamba3 (0.01 for parity, 0.05 for the other
  three tasks), not the value.
- `environment`: not stale as a statement of what ran, but incomplete (no Python version,
  commit or timestamp). The efficiency tracebacks show torch loaded from a global
  `Python314\site-packages` (log:231); the training logs contain no traceback, so the training
  interpreter is not recorded. The project venv today has Python 3.13.7 and torch 2.14.1+cu130,
  so a re-run will differ from the committed `torch` value.
- `peak_gpu_mb`: see the P6 table. P6 does not inflate any Transformer value, because the
  Transformer was the first model in every training process. Every s4d value, and the
  parity and selective-copy mamba2 values, are copies of the previous model's peak. The
  remaining values were set during their own run but still come from the pre-fix models where
  P1, P2 or D17 apply, and record allocated memory only.

### 5.1 P6: `peak_gpu_mb` against run order

`max_memory_allocated()` is a running maximum over the whole process when it is never reset,
so within one process the recorded values can only stay equal or rise in run order. They do,
in every process:

| Process | Order | Run (header line) | `peak_gpu_mb` | Previous value in the same process | Reading |
|---|---|---|---|---|---|
| parity | 1 | `parity_transformer` (log:1) | 109.5 | none | first model in its process: its own peak, not inherited |
| parity | 2 | `parity_s4d` (log:25) | 109.5 | 109.5 | equal to the previous model's value: inherited, own peak unknown (at most this value) |
| parity | 3 | `parity_mamba` (log:46) | 898.0 | 109.5 | higher than every earlier value: the process peak was set during this run |
| parity | 4 | `parity_mamba2` (log:67) | 898.0 | 898.0 | equal to the previous model's value: inherited, own peak unknown (at most this value) |
| parity | 5 | `parity_mamba3` (log:88) | 1701.3 | 898.0 | higher than every earlier value: the process peak was set during this run |
| selective copy | 1 | `selective_copy_transformer` (log:109) | 131.7 | none | first model in its process: its own peak, not inherited |
| selective copy | 2 | `selective_copy_s4d` (log:125) | 131.7 | 131.7 | equal to the previous model's value: inherited, own peak unknown (at most this value) |
| selective copy | 3 | `selective_copy_mamba` (log:138) | 513.2 | 131.7 | higher than every earlier value: the process peak was set during this run |
| selective copy | 4 | `selective_copy_mamba2` (log:151) | 513.2 | 513.2 | equal to the previous model's value: inherited, own peak unknown (at most this value) |
| selective copy | 5 | `selective_copy_mamba3` (log:164) | 1682.1 | 513.2 | higher than every earlier value: the process peak was set during this run |
| IMDb process 1 | 1 | `imdb_transformer` (log:177) | 1211.1 | none | first model in its process: its own peak, not inherited |
| IMDb process 1 | 2 | `imdb_s4d` (log:187) | 1211.1 | 1211.1 | equal to the previous model's value: inherited, own peak unknown (at most this value) |
| IMDb process 3 | 1 | `imdb_mamba2` (log:196) | 2865.0 | none | first model in its process: its own peak, not inherited |
| exact copy | 1 | `exact_copy_transformer` (log:804) | 518.5 | none | first model in its process: its own peak, not inherited |
| exact copy | 2 | `exact_copy_s4d` (log:814) | 518.5 | 518.5 | equal to the previous model's value: inherited, own peak unknown (at most this value) |
| exact copy | 3 | `exact_copy_mamba2` (log:821) | 1246.3 | 518.5 | higher than every earlier value: the process peak was set during this run |
| exact copy | 4 | `exact_copy_mamba` (log:828) | 3050.4 | 1246.3 | higher than every earlier value: the process peak was set during this run |

## 6. D17: every Mamba-2 run used the chunked scan

- No training config sets `scan_mode` (`git grep scan_mode` over `configs/` at b327a35,
  246f4cb and e0aea84 finds nothing), and the committed `model.kwargs` of the four mamba2
  training files have no `scan_mode` key.
- The trainer passes the config kwargs unchanged: `build_model` calls
  `build_backbone(mcfg["name"], mcfg.get("kwargs", {}))` (`trainer.py:60` @ b327a35, `:64` @
  246f4cb), and `build_backbone` forwards only the config keys (`models/__init__.py:41-46` @
  b327a35).
- `Mamba2Backbone` defaults to `scan_mode: str = "chunked"` (`mamba2_minimal.py:237` @ b327a35,
  `:243` @ bcc3f7d and 246f4cb), as does `Mamba2Block` (`:52`). `forward` picks
  `self.chunked_scan` when `scan_mode == "chunked"` (`:220` @ b327a35, `:226` @ bcc3f7d).
- The efficiency benchmark sets it explicitly (`DEFAULT_KWARGS["mamba2"]` has
  `scan_mode="chunked"`, `efficiency.py:37` @ b327a35, `:39` @ bcc3f7d), and all six mamba2
  entries record `"scan_mode": "chunked"` in `model_kwargs`.
- Efficiency run 2 crashed inside `chunked_scan` (log:304-305).

So D17 applies to all four mamba2 training files and all six mamba2 efficiency entries.

## 7. `results/efficiency_gpu.json`

### 7.1 What the file says about itself

- `environment`: `{"date": "2026-09-16", "device": "cuda", "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU", "torch": "2.14.0+cu130"}`.
  The benchmark rebuilt this block on every start (`efficiency.py:206-212` @ bcc3f7d), so it
  describes the last of the eight processes only.
- `config` equals `configs/efficiency.yaml`, which has not changed since b327a35 (batch 8,
  5 repeats, lengths 1024 to 32768, `model_length_limits` mamba 4096 and mamba3 4096).
- 24 entries. No entry carries a commit, timestamp or environment of its own.

### 7.2 P5: eight processes and several code versions

Each run below is one process. Each restart reloaded the file and skipped what was already in
it ("skipping ... (already measured)").

| Run | Log lines | Entries it added (# from the tables below) | How it ended | Code it ran |
|---|---|---|---|---|
| 1 | 203-254 | 0-5 (transformer 1024 to 32768), 6 (s4d 1024) | `RuntimeError: The size of tensor a (256) must match the size of tensor b (512)` in the S4D FFT convolution at s4d 2048 (log:211-254) | `efficiency.py` exactly as b327a35: frames 210, 126, 47, 116 (log:219-229) match `run_efficiency`, `benchmark_single`, `_time_gpu`, `train_step` @ b327a35, and there are no "skipping" lines (b327a35 had no resume). `s4_minimal.py` as b327a35, with the kernel chunks joined by `torch.cat(ks, dim=0)` (`:124` @ b327a35), the bug behind the crash |
| 2 | 255-309 | 7-11 (s4d 2048 to 32768), 12-14 (mamba2 1024 to 4096) | uncaught `torch.AcceleratorError: CUDA error: out of memory` in `chunked_scan` at mamba2 8192 (log:270-307) | uncommitted: resume is present, frames 128, 49, 118 match bcc3f7d but `run_efficiency` is at 227 (228 @ bcc3f7d) and `AcceleratorError` is not caught. `s4_minimal.py` already had the `dim=1` fix committed in bcc3f7d (s4d ran to 32768 without the crash). `mamba2_minimal.py` was still b327a35's: frames 254, 221, 206 (log:295-305) are `x = block(x)`, `y = scan(...)` and `S = cs_end[...]` @ b327a35, while @ bcc3f7d those statements sit at 260, 227, 212 |
| 3 | 310-404 | 15-16 (mamba2 8192, 16384), 17 (mamba2 32768 `oom_train`) | at mamba 1024 an out-of-memory error, then "Sticky error detected" and a second error from `torch.cuda.empty_cache()` in both handlers (log:331-402) | uncommitted: `benchmark_single` now catches `AcceleratorError` (frame 137 = bcc3f7d), `run_efficiency` frames 227 and 232. Which `mamba2_minimal.py` ran is not shown; the chunk grouping in bcc3f7d is commented "~1GB less transient memory at 32K" (inferred to be in place by this run) |
| 4 | 405-503 | 18-20 (mamba 1024 to 4096), 21 (mamba3 1024) | sticky CUDA error at mamba3 2048 (log:427-501) | uncommitted. The executed frames are 227 and 232 as in run 3, but the source text printed for them (log:461-467, 491-493) is a different version of the file: `efficiency.py` was edited on disk while this run was in progress |
| 5 | 504-599 | none | sticky CUDA error at mamba3 2048 (log:526-597) | uncommitted: frames 228, 137, 233 |
| 6 | 600-682 | 22 (mamba3 2048 `oom_train`) | sticky CUDA error inside `torch.manual_seed` at mamba3 4096 (log:623-680) | uncommitted: frames 228, 99, 235; `torch.cuda.empty_cache()` at 235 is not yet guarded |
| 7 | 683-779 | none | sticky CUDA error at mamba3 4096 (log:706-777) | uncommitted: frames 228, 137, 235 |
| 8 | 780-803 | 23 (mamba3 4096 `oom`, written by the outer handler, so no `model_kwargs` or `n_parameters`) | no traceback | fits bcc3f7d, where the outer handler wraps `empty_cache()` in its own `try` (`efficiency.py:231-238` @ bcc3f7d). Inferred: no traceback shows line numbers |

Consequences, beyond P1, P2, D17 and P16:

- The S4D kernel bug (`torch.cat(ks, dim=0)`) did not reach any committed number. s4d 1024
  (entry 6) was measured with it, but `S4DLayer.kernel` splits only above `chunk=1024`
  (`s4_minimal.py:110` @ b327a35), so T=1024 is one chunk. No training run went above 1024
  tokens either: parity uses 128, selective copy 64 and 128, IMDb `max_len: 512`, and exact
  copy 264 (`seq_len = 2 * max_eval + 8`, `synthetic.py:228` @ 246f4cb, with
  `eval_lengths` up to 128).
- The Transformer entries come from b327a35's benchmark: the old decode label and definition
  (section 7.5) and an `OutOfMemoryError`-only handler.
- Every `AcceleratorError` in the log reads "CUDA error: out of memory", so no committed
  OOM flag is shown to be a different CUDA error. The flags are still stale: they came from the
  headless train step (P16) and, for entries 17, 22 and 23, a CUDA context already in trouble.

### 7.3 Entries

Training-step fields. "run" refers to the table in 7.2.

| # | model | T | run | `train_step_ms` | `train_step_ms_std` | `train_tokens_per_s` | `peak_train_mb` | above 6140.5 MiB | flags | `n_parameters` |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | transformer | 1024 | 1 | 30.09 | 0.15 | 272237 | 419.0 |  |  | 1579520 |
| 1 | transformer | 2048 | 1 | 81.91 | 2.13 | 200035 | 724.6 |  |  | 1579520 |
| 2 | transformer | 4096 | 1 | 250.31 | 4.71 | 130911 | 1333.9 |  |  | 1579520 |
| 3 | transformer | 8192 | 1 | 846.56 | 11.64 | 77415 | 2552.4 |  |  | 1579520 |
| 4 | transformer | 16384 | 1 | 4482.01 | 16.74 | 29244 | 4989.4 |  |  | 1579520 |
| 5 | transformer | 32768 | 1 | 186025.38 | 1639.98 | 1409 | 9879.4 | yes |  | 1579520 |
| 6 | s4d | 1024 | 1 | 19.51 | 0.36 | 419908 | 293.7 |  |  | 182784 |
| 7 | s4d | 2048 | 2 | 43.44 | 1.01 | 377175 | 567.9 |  |  | 182784 |
| 8 | s4d | 4096 | 2 | 84.89 | 0.68 | 386005 | 1116.5 |  |  | 182784 |
| 9 | s4d | 8192 | 2 | 169.6 | 0.43 | 386412 | 2213.5 |  |  | 182784 |
| 10 | s4d | 16384 | 2 | 741.04 | 1.48 | 176876 | 4919.5 |  |  | 182784 |
| 11 | s4d | 32768 | 2 | 13456.29 | 136.15 | 19481 | 9819.7 | yes |  | 182784 |
| 12 | mamba2 | 1024 | 2 | 78.54 | 0.16 | 104302 | 754.2 |  |  | 875808 |
| 13 | mamba2 | 2048 | 2 | 222.66 | 5.42 | 73583 | 1484.2 |  |  | 875808 |
| 14 | mamba2 | 4096 | 2 | 842.33 | 14.83 | 38902 | 2939.0 |  |  | 875808 |
| 15 | mamba2 | 8192 | 3 | 8953.79 | 69.13 | 7319 | 5852.1 |  |  | 875808 |
| 16 | mamba2 | 16384 | 3 | 130426.11 | 429.54 | 1005 | 11678.2 | yes |  | 875808 |
| 17 | mamba2 | 32768 | 3 |  |  |  |  |  | oom_train | 875808 |
| 18 | mamba | 1024 | 4 | 2632.3 | 25.98 | 3112 | 1999.7 |  |  | 876544 |
| 19 | mamba | 2048 | 4 | 10726.96 | 15.54 | 1527 | 3970.9 |  |  | 876544 |
| 20 | mamba | 4096 | 4 | 248138.47 | 2030.59 | 132 | 7912.6 | yes |  | 876544 |
| 21 | mamba3 | 1024 | 4 | 10880.38 | 60.51 | 753 | 6646.7 | yes |  | 877872 |
| 22 | mamba3 | 2048 | 6 |  |  |  |  |  | oom_train | 877872 |
| 23 | mamba3 | 4096 | 8 |  |  |  |  |  | oom |  |

Inference and decode fields.

| # | model | T | `infer_ms` | `peak_infer_mb` | `decode_mode` | `decode_ms` | `decode_per_token_ms` |
|---|---|---|---|---|---|---|---|
| 0 | transformer | 1024 | 1.4 | 104.4 | `full-prefix recompute (no KV cache)` |  | 0.0015 |
| 1 | transformer | 2048 | 3.35 | 116.4 | `full-prefix recompute (no KV cache)` |  | 0.0016 |
| 2 | transformer | 4096 | 9.09 | 140.4 | `full-prefix recompute (no KV cache)` |  | 0.0022 |
| 3 | transformer | 8192 | 27.82 | 188.4 | `full-prefix recompute (no KV cache)` |  | 0.0034 |
| 4 | transformer | 16384 | 96.06 | 284.5 | `full-prefix recompute (no KV cache)` |  | 0.0059 |
| 5 | transformer | 32768 | 348.41 | 476.6 | `full-prefix recompute (no KV cache)` |  | 0.0107 |
| 6 | s4d | 1024 | 3.04 | 84.9 | `stepwise (O(1) state)` | 1769.9 | 1.7284 |
| 7 | s4d | 2048 | 5.37 | 121.9 | `stepwise (O(1) state)` | 4532.04 | 2.2129 |
| 8 | s4d | 4096 | 8.57 | 131.9 | `stepwise (O(1) state)` | 7490.58 | 1.8288 |
| 9 | s4d | 8192 | 17.12 | 152.0 | `stepwise (O(1) state)` | 15035.48 | 1.8354 |
| 10 | s4d | 16384 | 36.1 | 257.8 | `stepwise (O(1) state)` | 26460.02 | 1.615 |
| 11 | s4d | 32768 | 80.83 | 562.0 | `stepwise (O(1) state)` | 52512.41 | 1.6026 |
| 12 | mamba2 | 1024 | 7.39 | 47.1 | `sequential scan over full sequence (per-token = time/T)` |  | 0.0069 |
| 13 | mamba2 | 2048 | 13.59 | 71.2 | `sequential scan over full sequence (per-token = time/T)` |  | 0.0059 |
| 14 | mamba2 | 4096 | 22.66 | 119.4 | `sequential scan over full sequence (per-token = time/T)` |  | 0.0055 |
| 15 | mamba2 | 8192 | 54.09 | 164.2 | `sequential scan over full sequence (per-token = time/T)` |  | 0.0072 |
| 16 | mamba2 | 16384 | 104.5 | 293.0 | `sequential scan over full sequence (per-token = time/T)` |  | 0.0069 |
| 17 | mamba2 | 32768 |  |  |  |  |  |
| 18 | mamba | 1024 | 358.21 | 41.3 | `stepwise (O(1) state)` | 1422.7 | 1.3894 |
| 19 | mamba | 2048 | 712.67 | 59.5 | `stepwise (O(1) state)` | 2756.27 | 1.3458 |
| 20 | mamba | 4096 | 1903.36 | 95.9 | `stepwise (O(1) state)` | 7898.99 | 1.9285 |
| 21 | mamba3 | 1024 | 1169.8 | 39.4 | `sequential scan over full sequence (per-token = time/T)` |  | 1.2106 |
| 22 | mamba3 | 2048 |  |  |  |  |  |
| 23 | mamba3 | 4096 |  |  |  |  |  |

Staleness by field and model:

| Field | transformer | s4d | mamba2 | mamba | mamba3 |
|---|---|---|---|---|---|
| `train_step_ms`, `train_step_ms_std`, `train_tokens_per_s`, `peak_train_mb`, `oom_train`, `oom` | P16 | P16, P2 | P16, P1, D17 | P16 | P16, P1 |
| `infer_ms`, `peak_infer_mb` | no fix changes them (run 1, b327a35 code) | P2 | P1, D17 | no fix changes them (run 4) | P1 |
| `decode_mode`, `decode_ms`, `decode_per_token_ms` | P3 (old label, forward time / T) | P17, P2 | P3 (prefill time / T, labeled as a sequential scan), P1, D17 | no fix changes them (stepwise `step()`, run 4) | P3 (prefill time / T), P1 |
| `n_parameters`, `model_kwargs` | not stale | not stale | not stale | not stale | not stale (absent on entry 23) |
| any field, entries at P4 | entry 5 | entry 11 | entry 16 | entry 20 | entry 21 |

Every entry is also stale through P5: none records the code that produced it, and the fixed
benchmark refuses to resume from them.

### 7.4 P4: entries above the device total

The device total is 6438780928 bytes, 6140.5 MiB (log:209). These entries report a training
peak above it. No `peak_infer_mb` is above it.

| # | model | T | `peak_train_mb` | run |
|---|---|---|---|---|
| 5 | transformer | 32768 | 9879.4 | 1 |
| 11 | s4d | 32768 | 9819.7 | 2 |
| 16 | mamba2 | 16384 | 11678.2 | 3 |
| 20 | mamba | 4096 | 7912.6 | 4 |
| 21 | mamba3 | 1024 | 6646.7 | 4 |

Their timings include traffic to system memory and are not comparable with the other points.
Points below the total may also have spilled: reserved memory was not recorded.

### 7.5 P3 and P17: decode fields

- Transformer, entries 0-5: label `full-prefix recompute (no KV cache)` and
  `decode_per_token_ms = round(mean_ms / seq_len, 4)` where `mean_ms` is one no-grad forward
  over the whole sequence (`efficiency.py:172-181` @ b327a35). bcc3f7d changed the label to
  `full-prefix recompute per new token (no KV cache)` and the value to the full forward time
  (`efficiency.py:174-184` @ bcc3f7d), but these entries were never re-measured. The committed
  value times T reproduces `infer_ms` (for example 0.0107 x 32768 is about 350.6 against
  `infer_ms` 348.41).
- mamba2, entries 12-16, and mamba3, entry 21: label `sequential scan over full sequence
  (per-token = time/T)`, value = one no-grad forward over T tokens divided by T
  (`_decode_sequential`, `efficiency.py:81-85` and `:169-173` @ bcc3f7d). Neither model has a
  `step()` path, so this is prefill time divided by T, not decode. For mamba2 the forward is the
  chunked parallel scan, not a sequential one.
- The fixed benchmark (bd7d6f1) writes `prefill_per_token_ms` for mamba2 and mamba3 and keeps
  `decode_per_token_ms` for stepwise decode and for the Transformer's full forward per new
  token, so the two cannot be confused in new results.
- s4d, entries 6-11: real stepwise decode, but `step()` recomputed `_params()` and
  `zoh_discretization` on every token (`s4_minimal.py:142-143` @ b327a35), so `decode_ms` and
  `decode_per_token_ms` include that setup cost on every step.
- mamba, entries 18-20: stepwise decode; no listed fix changes it.

## 8. Figures: inputs and staleness

Every file in `figures/` and `reports/figures/` is drawn by `scripts/make_figures.py` from the
JSON files above, so each inherits the staleness of its inputs. They were regenerated in
54298dd from the same committed JSON (no number changed) with honest labels: the decode
figure plots only real stepwise decode and shows forward time / T separately, points that
exceeded GPU memory are ringed, chance lines follow the data generators, and figures built
from results without provenance carry a "pre-fix code" note. Before 54298dd no committed
script produced `reports/figures/*.pdf`; the table below describes what each figure plots.

| Figure | Reads | Stale through |
|---|---|---|
| `fig_parity.{png,pdf}`, `reports/figures/parity.pdf` | final `token_accuracy` of `parity_*.json` | S4D: P2, D10. Mamba-1: D10. Mamba-2: P1, D17. Mamba-3: P1, D10. Transformer bar: no fix, same pre-fix run |
| `fig_selective_copy.{png,pdf}`, `reports/figures/selcopy.pdf` | final `sequence_success`, `token_accuracy` of `selective_copy_*.json` | as parity |
| `fig_imdb_quality.{png,pdf}`, `reports/figures/imdb.pdf` | final `accuracy`, `auc` of `imdb_*.json` | S4D: P2, D10. Mamba-2: P1, D17. Transformer: no fix |
| `fig_learning_curves.{png,pdf}` | per-epoch IMDb `accuracy` and `train_loss`, selective-copy `sequence_success`, parity `token_accuracy` | as the three task figures |
| `fig_exact_copy.{png,pdf}` | `final_by_length` `token_accuracy` and `sequence_success` of `exact_copy_*.json` | S4D: P2, D10. Mamba-1: D10. Mamba-2: P1, D17. Transformer: no fix |
| `fig_task_formulation.{png,pdf}` | final selective-copy `sequence_success` and exact-copy `final_by_length["32"].sequence_success` | as its inputs |
| `fig_efficiency_latency.{png,pdf}` | `train_step_ms`, OOM flags | P16 for every model, plus P2 (s4d), P1 (mamba2, mamba3), D17 (mamba2), P4, P5 |
| `fig_efficiency_memory.{png,pdf}` | `peak_train_mb`, OOM flags | as latency |
| `fig_efficiency_throughput.{png,pdf}` | `train_tokens_per_s`, OOM flags | as latency |
| `fig_efficiency_decode.{png,pdf}` | `decode_per_token_ms` | P3 (transformer, mamba2, mamba3), P17 (s4d), P2, P1, D17, P5 |
| `reports/figures/efficiency.pdf` | train step time and peak GPU memory (axis text) | as latency and memory |

## 9. Re-running on the GPU

### 9.1 Before starting

Run these from the repository root.

PowerShell:

```powershell
git status --short                       # must print nothing: commit the fixes first
git log -1 --format="%H %s"              # the commit the new results will name
nvidia-smi                               # no other compute job (type C or C+G python) may be listed
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader   # must print nothing
Test-Path Env:CUDA_VISIBLE_DEVICES       # must print False
[Environment]::GetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "User")      # must print nothing
[Environment]::GetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "Machine")   # must print nothing
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue         # clears it for this shell if set
uv pip install --no-deps -e . --python .venv\Scripts\python.exe           # creates the new console scripts
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m pytest -q  # every test must pass
.\.venv\Scripts\ssmbench-train.exe --help
.\.venv\Scripts\ssmbench-efficiency.exe --help
```

Git Bash:

```bash
git status --short
git log -1 --format="%H %s"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader   # must print nothing
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then echo "CUDA_VISIBLE_DEVICES unset"; else echo "set to '${CUDA_VISIBLE_DEVICES}'"; fi
unset CUDA_VISIBLE_DEVICES               # an empty value is not the same as unset
uv pip install --no-deps -e . --python .venv/Scripts/python.exe
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/Scripts/python.exe -m pytest -q
```

Notes:

- The venv was made by uv and has no pip, hence `uv pip`. `--no-deps` reinstalls only this
  project in editable mode, which is what generates `ssmbench-train` and `ssmbench-efficiency`;
  no dependency changes.
- The IMDb run calls `load_dataset("imdb")` (`src/ssmbench/data/imdb.py`), which needs the
  Hugging Face cache or network access.
- Run one job at a time on the GPU. Do not run training and the efficiency sweep together.
- WDDM fallback: the five points in section 7.4 may spill to system memory again. The fixed
  benchmark records a spill flag for such points. Turning the fallback off (NVIDIA Control
  Panel, "CUDA - Sysmem Fallback Policy") would make them fail as out of memory instead; that is
  a driver setting and the owner's choice.

### 9.2 Training, one process per config

`--device cuda` makes a run fail if CUDA is not visible instead of quietly training on the CPU
(`device: auto` falls back to CPU, `src/ssmbench/utils.py:25-29`). Training overwrites
`results/<name>.json` and has no resume, so nothing needs deleting first.

PowerShell:

```powershell
.\.venv\Scripts\ssmbench-train.exe --config configs/parity.yaml --out results --device cuda
.\.venv\Scripts\ssmbench-train.exe --config configs/selective_copy.yaml --out results --device cuda
.\.venv\Scripts\ssmbench-train.exe --config configs/imdb.yaml --out results --device cuda
.\.venv\Scripts\ssmbench-train.exe --config configs/exact_copy.yaml --out results --device cuda
```

Git Bash:

```bash
.venv/Scripts/ssmbench-train.exe --config configs/parity.yaml --out results --device cuda
.venv/Scripts/ssmbench-train.exe --config configs/selective_copy.yaml --out results --device cuda
.venv/Scripts/ssmbench-train.exe --config configs/imdb.yaml --out results --device cuda
.venv/Scripts/ssmbench-train.exe --config configs/exact_copy.yaml --out results --device cuda
```

To keep a record of restarts, append the console output to a log, for example
`... 2>&1 | Tee-Object -FilePath rerun.log -Append` in PowerShell or `... 2>&1 | tee -a rerun.log`
in Git Bash.

| Config | Models it runs | Models it intentionally does not run | Files written | Previous `wall_time_seconds`, summed |
|---|---|---|---|---|
| `configs/parity.yaml` | transformer, s4d, mamba, mamba2, mamba3 | none | 5 | 628.3 |
| `configs/selective_copy.yaml` | transformer, s4d, mamba, mamba2, mamba3 | none | 5 | 1088.6 |
| `configs/imdb.yaml` | transformer, s4d, mamba2 | mamba, mamba3 (`configs/imdb.yaml:25-28`: their Python-loop scans make the wall clock prohibitive on this GPU) | 3 | 332.7 |
| `configs/exact_copy.yaml` | transformer, s4d, mamba2, mamba | mamba3 (the models list stops at line 35; no reason is given in the config; `docs/literature/repeat_after_me_reproduction.md:35` lists the same four models) | 4 | 1144.8 |

The last column adds up the committed `wall_time_seconds` of each config's files (arithmetic on
the values in section 3); the fixed code will take a different time.

The original IMDb results came from three processes (section 4). A single
`ssmbench-train --config configs/imdb.yaml` run is the intended replacement: with peaks reset
per model (P6), running all three models in one process no longer mixes their memory numbers.
To re-run only part of a config, pass `--models`, for example `--models mamba2`.

All four configs should be re-run in full, Transformer included: the Transformer metrics are
not changed by any fix, but the committed files record no code version.

### 9.3 Efficiency sweep

The fixed benchmark refuses to resume from entries whose source hash is missing or different,
and every committed entry has none. Remove the stale file first. Do not use `--force` for this:
it would merge the stale entries into the new file.

PowerShell:

```powershell
Remove-Item -LiteralPath results\efficiency_gpu.json
.\.venv\Scripts\ssmbench-efficiency.exe --config configs/efficiency.yaml
```

Git Bash:

```bash
rm -- results/efficiency_gpu.json
.venv/Scripts/ssmbench-efficiency.exe --config configs/efficiency.yaml
```

The committed version stays in history: `git show bcc3f7d:results/efficiency_gpu.json`.

If the sweep stops partway, run the same command again without changing any source file,
`batch_size`, `repeats`, torch build or GPU in between: resume reuses an entry only when all of
those match. If anything has to change, delete the file again and start over. `configs/efficiency.yaml` has `device: auto`, so check the new
file's environment block says CUDA before using it.

For planning only: in the old sweep the slowest training-step points were mamba 4096
(248138.47 ms per step), transformer 32768 (186025.38 ms) and mamba2 16384 (130426.11 ms), each
run as 2 warm-up plus 5 timed steps (`_time_gpu(..., warmup: int = 2)`, `efficiency.py:46`;
`repeats: 5`, `configs/efficiency.yaml:6`). The fixed code will differ.

### 9.4 Regenerating what depends on the results

PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/make_figures.py
.\.venv\Scripts\python.exe demo-site/build.py
Set-Location reports
latexmk -pdf Efficient_Sequence_Modeling_S4_to_Mamba3.tex
latexmk -c
Set-Location ..
```

Git Bash:

```bash
.venv/Scripts/python.exe scripts/make_figures.py
.venv/Scripts/python.exe demo-site/build.py
(cd reports && latexmk -pdf Efficient_Sequence_Modeling_S4_to_Mamba3.tex && latexmk -c)
```

- `scripts/make_figures.py` reads `results/` and writes `figures/fig_*.png` and `.pdf` and the
  four `reports/figures/*.pdf` the report includes (defaults `--results results --out figures
  --report-out reports/figures`).
- `demo-site/build.py` writes `demo-site/data.js` from `results/`.
- The report's tables and numbers are typed into the `.tex`. Update them before running
  `latexmk`, or the rebuilt PDF will still show the stale numbers.
- `.gitignore` does not exclude LaTeX build files; `latexmk -c` removes them and keeps the PDF.
- The README and docs also quote numbers by hand.

### 9.5 After the re-run

Replace this file's tables with the new runs: the commit and source hash each result names, the
date, the environment, and the new `peak_gpu_mb`, reserved-memory and spill fields. Keep the
sections on the old files, or point to this version in history, so the stale numbers quoted
elsewhere can still be traced.
