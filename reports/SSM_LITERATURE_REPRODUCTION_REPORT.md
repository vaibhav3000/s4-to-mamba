# SSM Literature Reproduction Report

Date: 2026-09-18. Extension selected from the Awesome-state-space-models
collection: **Repeat After Me** (Jelassi et al., ICML 2024, arXiv:2402.01032).

## Original paper claim

Fixed-size-state SSMs are fundamentally capped at copying: attention can
re-read any prompt position (a two-layer Transformer can in principle copy
exponentially long strings), while an SSM must compress the history into its
state. Synthetic experiments (strings over a 26-letter alphabet, `$s|answer.`
format) and pretrained-LLM evaluations support the claim.

## Our setup

Smallest meaningful reproduction, integrated into this repository:
- New task `ExactCopyDataset` (src/ssmbench/data/synthetic.py), faithful to
  the paper's formulation: `<bos> <string> <sep>` prompt, emit
  `<string> <eos>`; loss only on the answer; 26-letter alphabet.
- One training run per model: strings of length 5..32, 8000 samples, 6 epochs
  (48k example presentations), batch 64, lr 1e-3 cosine, seed 42.
- The SAME trained model evaluated at string lengths 32 (in-distribution),
  64 and 128 (extrapolation), fixed 264-token window, 256 samples per length.
- Models at matched budget (d_model 128, 3 blocks): Transformer 603k params,
  S4D 96k, Mamba-2 364k, Mamba-1 358k.
- Differences from the paper: ~1/8 model scale, our educational
  implementations, sinusoidal (not RoPE/ALiBi) Transformer PE, unpacked
  single-example contexts, one seed. Classification: **qualitative
  reproduction**, not exact.

## Results table (task-dependent behavior; NOT a universal ranking)

| Experiment | Task | Train Len | Test Len | Transformer | SSM (best of family) | Winner | Interpretation |
|---|---|---|---|---|---|---|---|
| selective_copy (ours) | keep k pairs, filter noise | 64 | 128 | 0.000 seq | 0.183 seq (mamba2) | SSM | selectivity is the bottleneck; LTI collapses (s4d 0.064 token) |
| exact_copy (reproduction) | reproduce full string | 5..32 | 32 | 1.000 seq | 0.000 seq (mamba2/mamba) | Transformer | state capacity is the bottleneck; matches the paper's claim |
| exact_copy (reproduction) | reproduce full string | 5..32 | 64 | 0.000 seq (0.036 tok) | 0.000 seq | nobody | extrapolation: absolute-PE Transformer collapses (paper's PE-dependence), SSMs capacity-bound |
| exact_copy (reproduction) | reproduce full string | 5..32 | 128 | 0.000 seq (0.032 tok) | 0.000 seq | nobody | same; chance token accuracy = 0.033 |
| parity (ours) | state tracking (XOR) | 128 | 128 | 0.000 seq (0.533 tok) | 0.420 seq (mamba3) | complex SSM | rotations required; real transitions fail |

## Findings

1. The paper's in-distribution ranking **reproduces at matched budget**:
   Transformer 1.000 vs Mamba-1/2 at 0.000 sequence success, 0.106-0.150
   token accuracy.
2. Gated-LTI S4D also solves in-distribution copying (1.000): an LTI kernel
   plus input gate implements an echo bank within trained lengths - then
   collapses to chance at L=64+. In-distribution copying does not require
   selectivity; length generalization does require capacity.
3. Extrapolation collapses for every model; the Transformer's collapse
   reproduces the paper's positional-encoding-dependence (their NoPE-style
   models fail the same way; their ALiBi variant extrapolated). Adding an
   ALiBi arm is documented future work.
4. Combined with our selective-copy benchmark, the ranking **flips with task
   formulation** - the core research question of this extension. Our results
   are benchmark-specific, and the repository now says so explicitly.

## Failure modes and honesty

- Mamba-1/2 learn slowly at our budget (0.15/0.11 token accuracy in-dist,
  still rising at schedule end); one shared budget was kept for all models
  (no per-model tuning), mirroring the paper's fixed protocol.
- Single seed; no error bars.
- No claim of exact reproduction: models, scale, encodings, and packing all
  differ from the paper.
- Nothing was tuned to force agreement with the paper; the s4d in-dist
  success (which the paper's framing would not have predicted) is reported
  as measured and explained mechanistically.

## Conclusion

The Transformer-vs-SSM copying literature and our selective-copy benchmark
are not in conflict: they measure different bottlenecks (state capacity vs
selectivity). The repository now treats all architecture-vs-architecture
results as task-dependent, with four task regimes measured (selective copy,
exact copy, parity/state tracking, IMDb classification).

Reproduction artifacts: configs/exact_copy.yaml, results/exact_copy_*.json,
figures/fig_exact_copy.png, figures/fig_task_formulation.png,
docs/copy_task_comparison.md, docs/literature/repeat_after_me_reproduction.md,
docs/literature/interview_defense.md.
