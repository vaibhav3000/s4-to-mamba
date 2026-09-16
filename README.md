# Efficient Long-Context Sequence Modeling: From S4 to Mamba-3

An experimental study of how efficient sequence architectures evolved from
S4 through Mamba, Mamba-2 and Mamba-3, and what trade-offs emerge in quality,
latency, memory and state-tracking capability. Everything is implemented in
pure PyTorch from first principles, every number is measured on this machine
(RTX 4050 Laptop, 6 GB), and every figure regenerates from committed JSON.

**Headline result.** On cumulative-parity tracking, the Mamba-3 recurrence
(complex-valued transitions) reaches **98.3% token accuracy** while Mamba-2,
Mamba-1, S4D and a Transformer all remain at chance (50.6-56.3%) - exactly the
capability gap the Mamba-3 paper predicts for real-transition SSMs. On
selective copying, LTI S4D collapses (**6.4%**) where selective SSMs reach
**67%**.

## Why this matters

Efficient sequence modeling is the main alternative to quadratic attention for
long context. Understanding it requires more than importing `mamba-ssm`: the
interesting questions - what is the state, what makes Mamba selective, why did
Mamba-2 move to a matmul form, what does complexity in Mamba-3 buy - are only
answerable if the recurrence is written down and tested. This repository is
that exercise, with benchmarks attached.

## The five implementations (src/ssmbench/models/)

| model | transition A | discretization | training form | decode |
|---|---|---|---|---|
| `transformer_minimal` | - | - | causal SDPA attention | full-prefix recompute (no KV cache, labeled) |
| `s4_minimal` (S4D) | complex diagonal, HiPPO-style init | ZOH with exact (e^z-1)/z | FFT convolution, O(T log T) | O(1)-state complex `step()` |
| `mamba_minimal` | real negative diagonal, input-dependent | exponential-Euler: exp(dA), dB | python-loop selective scan | true incremental `step()`, O(1) state |
| `mamba2_minimal` | scalar-per-head x I | exponential-Euler | **chunked SSD** matmuls + carried (N,P) state | sequential scan (per-token cost) |
| `mamba3_minimal` | **complex** diagonal, input-dependent | **exponential-trapezoidal** (data-dependent lambda) | python-loop selective scan | sequential scan (per-token cost) |

Correctness is not asserted, it is tested (`tests/`, 14 tests): the S4D
convolution form equals its recurrent step form; Mamba-2's chunked SSD equals
its sequential reference; Mamba-3 reduces to the Mamba-1/2 update when
lambda=1 and the rotation frequency is zero; the decode path equals the
parallel path.

## Experiments and results (results/*.json)

### 1. Parity (state tracking) - the Mamba-3 motivation, reproduced

Cumulative XOR over L=128; the target at position t is the parity of inputs
1..t. Real non-negative transitions can only decay; they cannot rotate state,
which parity requires. 20 epochs, matched small budgets:

| model | token accuracy | full-sequence success |
|---|---|---|
| **mamba3** (complex) | **0.983** | **0.42** |
| mamba2 | 0.563 | 0.0 |
| mamba | 0.559 | 0.0 |
| transformer | 0.533 | 0.0 |
| s4d | 0.506 | 0.0 |

S4D has complex transitions but LTI ones (fixed rotation, not input-dependent),
which is why it also fails - the rotation angle must respond to each bit.

### 2. Selective copying - selectivity, demonstrated

Marker-value pairs embedded in noise; the model must emit the remembered
values after a separator. Train L=64 (3-5 pairs), test L=128 (5-10 pairs),
12 epochs:

| model | sequence success | token accuracy |
|---|---|---|
| mamba2 | **0.183** | **0.670** |
| mamba | 0.149 | 0.649 |
| mamba3 | 0.129 | 0.215 |
| transformer | 0.0 | 0.205 |
| s4d | 0.0 | 0.064 |

LTI S4D cannot gate noise out of its state and fails outright; the selective
SSMs store and release markers. mamba3 underperforms mamba1/2 here - its
rotational dynamics appear to trade recall sharpness for state-tracking power
at this tiny budget - reported as measured.

### 3. IMDb sentiment (quality on real text)

Same protocol for every model: 12.5k training samples (fixed seed), max length
512, 3 epochs, d_model=128, 4 blocks, cosine schedule:

| model | accuracy | AUC |
|---|---|---|
| s4d | **0.827** | **0.907** |
| transformer | 0.813 | 0.894 |
| mamba2 | see `results/imdb_mamba2.json` | |

The python-loop sequential implementations (mamba, mamba3) are deliberately
not trained on IMDb at this scale: their wall-clock is dominated by kernel
launch overhead (see below). The selective-SSM family is represented by
mamba2, which shares the recurrence and differs only in the transition
parameterization.

### 4. Efficiency vs sequence length (1K-32K, RTX 4050)

`results/efficiency_gpu.json` holds per-model train-step latency, peak memory,
throughput and decode cost, CUDA-event timed, 5 repeats. Educational sequential
scans are capped at 4K length: their wall-clock growth with T is itself the
documented motivation for fused kernels. Figures are regenerated with
`python scripts/make_figures.py`.

### 5. The wall-clock lesson (the honest part)

The minimal Mamba-1/3 scans are O(T) in operations but slow in wall-clock:
a python loop over T steps x n_layers launches thousands of tiny kernels and
builds a T-step autograd graph. The chunked Mamba-2 form cuts the loop by
`chunk_size` and turns the inner work into tensor-core matmuls; the official
repository goes further with fused Triton/CUDA kernels (Linux-only, not used
here). Theoretical complexity and wall-clock performance are different claims,
and this repository measures both.

## What is NOT implemented (and why that is stated)

- Full S4 (NPLR parameterization, Woodbury-corrected Cauchy kernel): S4D keeps
  every idea this study needs; the delta is documented in docs/evolution.md.
- Fused/official kernels: this repo is dependency-light pure PyTorch; the
  official `mamba-ssm` requires Linux and CUDA compilation.
- Mamba-3's MIMO formulation and its chunked dual: the recurrence, complex
  states and trapezoidal discretization are implemented; the parallel form is
  described in docs/evolution.md.

## Reproducibility

```bash
python -m pip install -e ".[dev]"        # torch required (CPU or CUDA)
python -m pytest tests/ -q               # 14 correctness tests
python scripts/run_training.py --config configs/parity.yaml --out results
python scripts/run_training.py --config configs/selective_copy.yaml --out results
python scripts/run_training.py --config configs/imdb.yaml --out results
python scripts/run_efficiency.py --config configs/efficiency.yaml
python scripts/make_figures.py           # figures/ from committed JSON
```

Configs pin seeds, protocols and model sizes; results are committed so figures
can be rebuilt without a GPU.

## Project structure

```
src/ssmbench/models/     five backbones + shared blocks + registry
src/ssmbench/data/       IMDb pipeline; selective copy; parity
src/ssmbench/train/      config-driven trainer, JSON results
src/ssmbench/benchmarks/ sequence-length efficiency harness
configs/                 experiment YAMLs
tests/                   equivalence + shape + determinism tests
results/                 committed experiment outputs (JSON)
figures/                 generated plots (PNG + PDF)
docs/                    evolution.md, interview_guide.md
legacy/                  the original project's scripts, preserved with notes
```

## References

See REFERENCES.md. Primary: S4 (arXiv 2111.00396), S4D (2206.11893), HiPPO
(2008.07669), Mamba (2312.00752), Mamba-2 (2405.21060), Mamba-3 (2603.15569).
The official state-spaces/mamba repository (Apache-2.0) was consulted for
parameterization conventions; no code was copied.

## License

MIT.
