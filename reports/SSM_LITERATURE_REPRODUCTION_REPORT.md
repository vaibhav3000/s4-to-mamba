# SSM Literature Reproduction Report

> **Status (November 2026).** This note was written for the earlier single-seed version
> of this repository, whose models had defects that have since been fixed. The numbers and
> result files it quotes are kept for the record (the files are in the git history before
> commit `ab29379`) and are superseded by the three-seed results in `results/SUMMARY.md`,
> the README and the report.

Date: 2026-09-18. Extension selected from the Awesome-state-space-models
collection: **Repeat After Me** (Jelassi et al., ICML 2024, arXiv:2402.01032).

Status: the results below come from single-seed runs made before later fixes to the models
(the S4D ZOH input matrix, how Mamba-2 and Mamba-3 heads read `B` and `C`, and weight decay
on SSM dynamics parameters). They are stale until re-run; see results/PROVENANCE.md. Where
the original text drew conclusions the runs do not support, it is qualified below.

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
- One training run per model: strings of length 32 only (every training string has
  exactly 32 letters), 8000 samples, 6 epochs (48k example presentations), batch 64,
  lr 1e-3 cosine, seed 42.
- The SAME trained model evaluated at string lengths 32 (in-distribution),
  64 and 128 (extrapolation), fixed 264-token window, 256 samples per length.
- Models at matched width and depth (d_model 128, 3 blocks), not matched parameter
  count: Transformer 603k params, S4D 96k, Mamba-2 364k, Mamba-1 358k.
- Differences from the paper: ~1/8 model scale, our educational
  implementations, sinusoidal (not RoPE/ALiBi) Transformer PE, unpacked
  single-example contexts, a single training string length, one seed.
  Classification: **qualitative reproduction**, not exact.

## Results table (task-dependent behavior; NOT a universal ranking)

| Experiment | Task | Train Len | Test Len | Transformer | SSM (best of family) | Winner | Interpretation |
|---|---|---|---|---|---|---|---|
| selective_copy (ours) | keep k pairs, filter noise | 64 (3-5 pairs) | 128 (5-10 pairs) | 0.000 seq | 0.183 seq (mamba2) | SSM | LTI collapses (s4d 0.064 token); the SSM score is out of distribution in length and pair count and sits near the score of recalling only the first five values |
| exact_copy (reproduction) | reproduce full string | 32 (fixed) | 32 | 1.000 seq | 1.000 seq (s4d); 0.000 seq (mamba2/mamba) | Transformer and S4D | a fixed-offset copy at one length, which LTI S4D also learns; Mamba-1/2 were still improving after 6 epochs, so state capacity is not shown to be the bottleneck |
| exact_copy (reproduction) | reproduce full string | 32 (fixed) | 64 | 0.000 seq (0.036 tok) | 0.000 seq | nobody | extrapolation from a single training length: the absolute-PE Transformer collapses (consistent with the paper's PE dependence) and so does every SSM |
| exact_copy (reproduction) | reproduce full string | 32 (fixed) | 128 | 0.000 seq (0.032 tok) | 0.000 seq | nobody | same; chance token accuracy is 1/30 = 0.033 over the full vocabulary, 1/26 over the letters |
| parity (ours) | state tracking (XOR) | 128 | 128 | 0.000 seq (0.533 tok) | 0.420 seq (mamba3) | input-dependent complex SSM | consistent with input-dependent rotation being needed: real selective Mamba-1/2 and complex LTI S4D stay near chance; one seed, no theta = 0 ablation |

## Findings

1. The in-distribution gap between the Transformer and Mamba-1/2 appears at
   matched width and depth: Transformer 1.000 vs Mamba-1/2 at 0.000 sequence
   success, 0.106-0.150 token accuracy. It is weak evidence for the paper's claim:
   training used a single string length, the Mamba-1/2 validation loss was still
   falling when the 6-epoch schedule ended, and the LTI S4D also reached 1.000.
2. Gated-LTI S4D also solves in-distribution copying (1.000). Because every
   training string has length 32, the answer is the prompt delayed by a fixed
   offset, which a fixed LTI kernel can implement; S4D then collapses to chance at
   L=64 and beyond. In-distribution copying at one length does not require
   selectivity. The runs do not show what length generalization requires, since no
   model was trained on more than one length.
3. Extrapolation collapses for every model. The Transformer's collapse is
   consistent with the paper's finding that copying length generalization depends
   on the positional encoding (their NoPE-style models fail the same way; their
   ALiBi variant extrapolated); we trained only one encoding, so we do not test it.
   Adding an ALiBi arm is documented future work.
4. Combined with our selective-copy benchmark, the ranking **differs with task
   formulation**, which is the core research question of this extension. The
   comparison is uneven: the selective-copy score is out of distribution (L=128 and
   5-10 pairs after training on L=64 and 3-5 pairs), while the exact-copy score that
   separates the models is in distribution at the single training length. Our
   results are benchmark-specific, and the repository says so explicitly.

## Failure modes and honesty

- Mamba-1/2 learn slowly at our training budget (0.11/0.15 token accuracy
  in-dist, still rising at schedule end); one shared training budget was kept for all
  models (no per-model tuning), mirroring the paper's fixed protocol. This slow
  learning is enough to explain their 0.000 at L=32; the runs do not show a capacity
  limit.
- Single seed; no error bars.
- No claim of exact reproduction: models, scale, encodings, and packing all
  differ from the paper.
- Nothing was tuned to force agreement with the paper; the s4d in-dist
  success (which the paper's framing would not have predicted) is reported
  as measured and explained mechanistically.

## Conclusion

The Transformer-vs-SSM copying literature and our selective-copy benchmark
are not in conflict: they are designed to measure different bottlenecks (state
capacity vs selectivity), although our exact-copy runs do not isolate state capacity
(Findings 1 and 2). The repository now treats all architecture-vs-architecture
results as task-dependent, with four task regimes measured (selective copy,
exact copy, parity/state tracking, IMDb classification).

Reproduction artifacts: configs/exact_copy.yaml, results/exact_copy_*.json,
figures/fig_exact_copy.png, figures/fig_task_formulation.png,
docs/copy_task_comparison.md, docs/literature/repeat_after_me_reproduction.md,
docs/literature/interview_defense.md. The committed figures are pre-fix:
figures/fig_exact_copy.png labels the training lengths "5 to 32", which is wrong (training
used length 32 only), and figures/fig_task_formulation.png sets selective copy at test
L=128 (out of distribution) against exact copy at the in-distribution L=32.
