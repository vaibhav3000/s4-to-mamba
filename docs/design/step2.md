# Step 2 design: fair baselines, inference systems, chunked Mamba-3, multi-seed results

Status: approved scope, written before implementation. Terms in **bold** are defined in
[docs/GLOSSARY.md](../GLOSSARY.md); decisions that are hard to reverse are recorded in
[docs/adr/](../adr/).

## 1. Where Step 1 left the project

Step 1 fixed two math bugs (S4D ZOH input matrix, Mamba-2/3 grouped B and C), made the scans
dtype-aware, added naive float64 references with gradchecks, and hardened the efficiency
benchmark (provenance, spill flags, OOM handling). Every committed number was produced by
pre-fix code and is marked stale in `results/PROVENANCE.md`. Nothing has been re-run.

What is still missing, and what this step adds:

| Gap | Effect today |
|---|---|
| One seed per run | No spread, so no claim can be separated from seed noise |
| Test split doubles as the per-epoch "val" metric (D12) | Monitoring touches the test set |
| Transformer baseline is fp32 SDPA with an unrecorded backend, no RoPE, no KV cache | The baseline is weaker than a 2026 practitioner would deploy, and its decode cost is a full-prefix recompute |
| Width-matched configs with unequal parameter counts | Accuracy differences mix architecture and size |
| No step() for Mamba-2 and Mamba-3 | Their decode cost cannot be measured |
| Mamba-3 has only the sequential scan | It is 10 s per training step at T = 1024 and cannot run long sequences |
| Ad hoc figures and hand-typed numbers in README and report | Numbers drift from the JSON |
| pip installs with an unpinned dependency set | Environments cannot be reproduced |

## 2. Scope

1. **Multi-seed re-run of every result** on the fixed code, 3 to 5 **run seeds** per **model
   variant**, reported as mean ± sample std with n. Tasks: parity (with a θ = 0 **rotation
   ablation** and a **length-extrapolation** evaluation), selective copy (in-distribution and
   out-of-distribution evaluation sets), exact copy trained at a fixed length and at variable
   lengths, IMDb, and the efficiency sweep.
2. **Fair baselines**: the **attention backend** is pinned per run and the backend that actually
   ran is recorded; a RoPE Transformer; **parameter-matched** configs. Naive attention stays only
   as a labelled **reference**.
3. **Inference systems**: a KV-cache Transformer decode and an O(1) **decode step** for every SSM,
   measured as **time to first token**, tokens/s, per-token latency and peak allocated and
   reserved VRAM for prompt lengths 1K to 32K; bf16/fp16 numerical stability against fp32 and
   float64 references; torch.compile graph-break checks.
4. **Chunked parallel form of Mamba-3** (complex SSD plus the trapezoidal **two-band correction**),
   verified against the recurrence in float64 with explicit tolerances and gradcheck.
5. **Packaging and report**: uv with a lockfile and CPU vs CUDA torch extras; figures and tables
   generated from committed JSON; the LaTeX report rewritten around multi-seed tables; demo data
   regenerated.

Not in scope: fused Triton or CUDA kernels, the full S4 (NPLR) layer, MIMO Mamba-3,
data-dependent θ, hyperparameter tuning, torch 2.15 (ADR 0007).

## 3. Model layer

### 3.1 One decode interface for every backbone

Every backbone gains the same three methods. The **decode state** is opaque to callers.

```python
state = backbone.init_state(batch_size, device, max_len)  # empty prompt
features, state = backbone.prefill(emb, max_len)  # parallel form over a prompt
features_t = backbone.step(emb_t, state)  # one token, mutates state
```

- `prefill` runs the **parallel form** (FFT convolution, chunked SSD, causal SDPA) and returns
  the state a recurrence would hold after the prompt. `max_len` sizes the Transformer's KV cache
  (prompt plus every token that will be generated); SSMs ignore it.
- `step` takes `emb_t` of shape (B, 1, D) and returns (B, 1, D).
- The contract tested for every backbone, in float64 at rtol = atol = 1e-10:
  `prefill(x[:, :k])` followed by `step` over `x[:, k:]` reproduces `forward(x)` position by
  position, for k in {0, 1, a chunk boundary, T - 1}, with T not a multiple of any chunk size.
- Block-level `init_state`/`step` on the Mamba-1 and S4D blocks keep their current signatures.

Per backbone:

| Backbone | Decode state per layer | `prefill` |
|---|---|---|
| Transformer | K and V caches (B, H, max_len, d_head), preallocated, dtype of K | causal SDPA, cache filled for positions < T |
| S4D | complex h (B, D, N) plus the cached discretization | FFT conv for outputs; h_T = sum_k Abar^(T-1-k) Bbar x_k as one chunked contraction |
| Mamba-1 | h (B, d_inner, N), conv history | sequential scan, final h |
| Mamba-2 | h (B, H, N, P), conv history | chunked SSD, final chunk state |
| Mamba-3 | complex h (B, H, N, P), previous input s = B x^T, conv history | chunked complex SSD, final state and boundary input |

The short causal convolution is the same in the three Mamba blocks; its prefill and step logic
moves into one helper in `models/blocks.py` instead of three copies.

### 3.2 Transformer: positions, backends, KV cache

`TransformerBackbone(pos_encoding="sinusoidal" | "rope", attn_backend="auto" | "flash" |
"efficient" | "cudnn" | "math" | "naive")`.

- RoPE rotates q and k per head by the absolute position (base 10000, rotate-half convention,
  d_head even). With RoPE there is no additive table.
- `attn_backend` other than `auto` wraps every SDPA call in `sdpa_kernel([backend])`, so a
  backend that cannot run raises instead of silently falling back. `naive` is an explicit
  softmax(QK^T / sqrt(d) + causal mask) V in the input dtype, kept as the reference oracle and
  labelled as such wherever it is measured.
- The backend that ran is detected from the profiler trace of one forward (the aten op names
  `_scaled_dot_product_{flash_attention, efficient_attention, cudnn_attention,
  attention_math}`; on CPU `_flash_attention_for_cpu`), not inferred (ADR 0004).
- A decode step attends one query to cache[:, :, :t+1] with no mask; prefill uses is_causal on
  a square score matrix, so the upper-left alignment of non-square causal masks never arises.

### 3.3 Mamba-2 step

h_t = exp(delta_t A_h) h_{t-1} + delta_t B_{g(h),t} x_t^T, y_t = C_t^T h_t + D x_t, then the
SiLU gate and output projection, exactly as `forward`. Verified against both scan modes.

### 3.4 Mamba-3: chunked form, step, rotation flag

New constructor arguments: `scan_mode="chunked" | "sequential"` (default chunked, as in
Mamba-2), `chunk_size`, and `rotation: bool = True`. With `rotation=False` θ is a zero buffer
rather than a parameter, so the transition is real and decay-only; this is the parity ablation.

**Derivation.** Per head h and state slot n, with alpha_t = exp(delta_t A_n),
A_n = -exp(a_n) + i θ_n, s_t = B_t x_t^T (real N x P), gamma_t = lambda_t delta_t and
beta_t = (1 - lambda_t) delta_t alpha_t:

    h_t = alpha_t h_{t-1} + beta_t s_{t-1} + gamma_t s_t,    y_t = C_t^T Re(h_t).

Unroll with L_{ts} = prod_{r=s+1..t} alpha_r = exp(A_n D_{ts}), where D_{ts} = sum_{r=s+1..t}
delta_r is a real segment sum of step sizes. Because L_{t,s+1} alpha_{s+1} = L_{ts}, the
complex factor alpha in beta folds into the decay, and the beta term becomes a real weight on the
previous position:

    h_t = sum_{s <= t} L_{ts} * s_s * W_{ts},
    W_{ts} = gamma_s + [s < t] beta'_s,    beta'_s = (1 - lambda_{s+1}) delta_{s+1}.

W is the **two-band correction**: the diagonal carries gamma and the band below it carries
beta'. Since B, C and x are real, Re(C^T L B) = sum_n C_n B_n Re(L_n) and

    y_t = sum_{s <= t} G_{ts} W_{ts} x_s,
    G_{ts} = sum_n C_{t,n} B_{s,n} exp(Re(A_n) D_{ts}) cos(θ_n D_{ts}).

This is Mamba-2's SSD with two changes: the decay depends on the state slot (so G sums over n
inside the decay rather than outside it) and the input weight is two-banded.

**Chunked algorithm.** Pad T to a multiple of the chunk size Q (padded delta = 0, x = 0).
Within each chunk:

- intra-chunk: D_{ij} as a direct masked cumsum (no difference of running cumsums, as in
  Step 1's Mamba-2 fix), G from (Q, Q, N) real decays, y_intra = (G * W * causal) x;
- carry-in: y_inter_i = Re(sum_n C_{i,n} exp(A_n cs_i) H_n), where cs_i is the inclusive prefix
  of delta in the chunk and H = S + beta'_{last} s_{last} is the previous chunk's final state
  plus its **boundary input**;
- carry-out: S_next = exp(A_n cs_end) H + sum_j exp(A_n suffix_j) W_{end,j} s_j.

The (Q, Q, N) decay tensor is materialized per group of chunks, with the group size chosen from
a byte budget so the transient stays bounded at 32K tokens (ADR 0002). Cost per token is
O(Q N + Q P) per head instead of the sequential scan's O(N P) work in T Python steps.

**Verification** (float64, tolerances stated in each test): chunked vs sequential outputs and
all parameter gradients at rtol = atol = 1e-10, across n_groups in {1, 2}, lambda in {0, 1/2,
1, learned}, θ in {0, rotating}, T in {1, Q - 1, Q, Q + 1, 5Q + 3}, batch 1 and 3; gradcheck of
the chunked scan (eps 1e-6, atol 1e-7, rtol 1e-5); the existing naive per-(head, slot)
reference through `forward` in both scan modes; float32 vs float64 at 1e-4 relative to output
scale; tiny and large decays; chunk-size invariance; bf16 autocast leaves the scan in float32.

## 4. Training layer

### 4.1 Model variants

A config's `models` mapping is keyed by **model variant** label. An optional `backbone:` key
names the **architecture**; without it the label is the architecture. Results are written per
label:

```yaml
models:
  transformer:       {d_model: 64, n_layers: 2, n_heads: 4}
  transformer_rope:  {backbone: transformer, pos_encoding: rope, d_model: 64, n_layers: 2, n_heads: 4}
  mamba3_no_rotation: {backbone: mamba3, rotation: false, d_model: 80, n_layers: 2, d_state: 32, headdim: 32}
```

### 4.2 Runs, seeds, evaluation sets

- A **run** is one (config, variant, run seed). The run seed sets initialization, dropout and
  data order; the datasets are fixed by the data seed and are identical across runs (ADR 0001).
- Every task has a **validation set** (training distribution, its own data seed) used for the
  per-epoch history, and one or more **test evaluation sets** scored once after the last epoch.
  No early stopping or checkpoint selection: the reported model is the final one.

| Task | Evaluation sets |
|---|---|
| parity | `T=128` (in distribution), `T=256`, `T=512`, `T=1024`, `T=2048` |
| selective copy | `id` (T = 64, 3 to 5 pairs), `ood` (T = 128, 5 to 10 pairs) |
| exact copy, fixed and variable length | `L=16`, `L=32`, `L=64`, `L=128` |
| IMDb | `test` (the 25k test split); validation is 2,500 training reviews held out from the training subset |

- Result files: `results/train/<config>/<variant>/seed<k>.json`, each with the resolved config,
  per-epoch validation history, final metrics per evaluation set, parameter count, attention
  backend (Transformers), wall time, provenance (git SHA, dirty flag, source hash, config hash,
  torch, CUDA, cuDNN, GPU, driver) and memory.
- `ssmbench-train --config C --seeds 0 1 2` loops over variants and seeds; a run whose result
  file has the same source and config hash is skipped, any other result file is refused unless
  `--force`. Each epoch writes a checkpoint (model, optimizer, scheduler, every RNG state) under
  `checkpoints/` (git-ignored); an interrupted run resumes from it and finishes bitwise identical
  to an uninterrupted one (tested on CPU).
- Training runs with `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA initializes, so a seed reproduces exactly.

### 4.3 Summaries

`ssmbench-summarize` (module `ssmbench.analysis.summary`) reads every run of a config and
writes `results/train/<config>/summary.json`: per variant and evaluation set, the mean, sample
std (ddof = 1), n, min, max and the per-seed values of each metric, the parameter count and the
chance level. It refuses to summarize runs whose source or config hashes differ.

### 4.4 Parameter matching

`scripts/match_params.py` picks, for every SSM variant, the d_model on its constraint grid
(d_inner divisible by headdim, n_heads by n_groups) whose total parameter count is closest to
the task's Transformer at its original width, with depth and every other setting fixed. The
chosen widths are written into the configs as literal numbers; a test rebuilds every model and
asserts each count is within 10% of the anchor (ADR 0005).

## 5. Benchmarks

All timing uses CUDA events after warm-up and `torch.cuda.synchronize()`, reports median and
p90 (mean and std kept for continuity), and records peak allocated and peak reserved memory
with the Step 1 spill flag. Every entry stores precision, requested and detected attention
backend, batch size, lengths and provenance; every benchmark resumes per point.

### 5.1 Efficiency sweep (`ssmbench-efficiency`, existing)

Training step (forward, backward, AdamW) and full forward at T = 1K to 32K, batch 8. New:

- `precision: fp32 | bf16` (bf16 = autocast; the SSM projections and scans stay float32);
- variants with `backbone:` as in 4.1, so `transformer_flash` (bf16), `transformer_cudnn`
  (bf16), `transformer_efficient` (fp32 and bf16), `transformer_math` and `transformer_naive`
  (references) are separate rows;
- `stop_after_spill`: after a variant's training step spills or runs out of memory, longer
  lengths of that variant are recorded as skipped instead of spending minutes per point in
  WDDM paging;
- the decode phase uses the new decode interface for every model (KV cache for the Transformer)
  and can be switched off with `measure_decode: false`; the dedicated decode numbers come from
  5.2.

### 5.2 Inference benchmark (`ssmbench-inference`, new)

For prompt lengths 1K to 32K, batch 1, precision fp32 and bf16: **time to first token** (prompt
on device to first generated token id: embedding, `prefill`, LM head on the last position,
argmax), then G = 64 greedy **decode steps** each timed with its own event pair (per-token median
and p90 over every step of every repeat; tokens/s = G / total decode time), peak allocated and
reserved memory over the session, and the analytic **decode state** size in bytes (KV cache
versus SSM state).

### 5.3 Numerics (`ssmbench-numerics`, new)

For every architecture at T in {1024, 4096} and 3 initialization seeds: relative L2 and max
absolute error of the backbone output under bf16 and fp16 autocast against fp32, and of fp32
against float64; the same for logits after a 256-step decode from a 1K prefill (error growth in
the recurrent state); non-finite counts. Pass criteria are not imposed on GPU results; the
numbers are reported.

### 5.4 Compile check (`ssmbench-compile`, new)

`torch._dynamo.explain` on `forward` (training mode) and `step` of every architecture: graph
count, graph-break count and break reasons; outputs of `torch.compile(backend="aot_eager")`
against eager at float32 tolerances. Forward graph breaks found here are fixed, and a test pins
the forward of every architecture at zero breaks. Inductor needs Triton, which the Windows
environment does not have; compiled timings are measured only where Triton exists (Linux or
WSL2), and the report says so.

## 6. Results, figures, tables

```
results/
  ENVIRONMENT.json                     uv lock hash, package versions, torch/CUDA/cuDNN, driver, GPU, OS
  train/<config>/<variant>/seed<k>.json
  train/<config>/summary.json
  efficiency/sweep_<precision>.json
  inference/decode.json
  numerics/numerics.json
  compile/compile.json
  logs/<job>.log                       console output of each GPU job
  PROVENANCE.md                        rewritten: which commit produced what
```

- `scripts/make_figures.py` reads only summaries and benchmark JSON and writes `figures/` (PNG
  and PDF) and `reports/figures/` (PDF); error bars are ± one sample std with n in the caption.
- `scripts/make_tables.py` writes `reports/tables/*.tex` and the generated blocks of
  `README.md` and `results/SUMMARY.md` between `<!-- generated:NAME -->` markers. A test checks
  the committed tables equal a fresh generation from the committed JSON, so no number is typed by
  hand (ADR 0006).
- `demo-site/build.py` reads the same summaries.
- The pre-fix JSON files are removed from the tree when their replacements land;
  `results/PROVENANCE.md` names the last commit that contains them.

## 7. Packaging

pyproject keeps `torch>=2.14,<2.15` as a dependency and adds extras `cpu` and `cu130`, both
`torch==2.14.1`, declared as conflicting, each bound to its PyTorch index (`explicit = true`).
Dev tools move to a `dev` dependency group. `uv.lock` is committed; CI runs
`uv sync --locked --extra cpu`; the Dockerfile installs from the same pins. The local `.venv`
is synced with `--extra cu130` (ADR 0007).

## 8. GPU run plan

Run only inside a window the owner grants, one job at a time, in the background, logged to
`results/logs/`, every job resumable. `scripts/run_gpu_plan.py` runs the list in order and
stops at the first failure.

| # | Job | Runs or points | Estimate |
|---|---|---|---|
| 0 | smoke: 2 steps of every variant on CUDA, determinism check | 30 | 5 min |
| 1 | parity, 7 variants x seeds | 21 (3 seeds) | 25 to 40 min |
| 2 | selective copy, 6 variants x seeds | 18 | 40 to 60 min |
| 3 | exact copy, fixed length, 6 variants x seeds | 18 | 60 to 90 min |
| 4 | exact copy, variable length | 18 | 60 to 90 min |
| 5 | IMDb, 5 variants x seeds | 15 | 40 to 60 min |
| 6 | efficiency sweep, fp32 and bf16 | about 80 points | 60 to 90 min |
| 7 | inference benchmark | about 70 points | 30 to 60 min |
| 8 | numerics and compile check | 2 jobs | 15 min |

About 6 to 9 hours with 3 seeds. Mamba-1's sequential scan dominates jobs 2 to 4; seeds 4 and 5
can be added per task afterwards because every run is independent. The estimates come from the
pre-fix wall times in `results/*.json` and are replaced by measured times in the report.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Deterministic mode rejects a CUDA op | the smoke job runs every variant for two steps with determinism on before any long job |
| WDDM paging makes long points take minutes | `stop_after_spill`, spill flags, per-point resume |
| The owner's own GPU job shares the card | nvidia-smi check before every job; `available_at_start` makes the spill flag honest |
| Mamba-3 chunked memory at 32K | byte-budgeted chunk groups, measured on CPU first at reduced size |
| A code change mid-campaign mixes versions | the summarizer refuses runs with different source hashes |
