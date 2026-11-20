# SSM Literature Reproduction Report

Extension selected from the Awesome-state-space-models collection: **Repeat After Me**
(Jelassi et al., ICML 2024, arXiv:2402.01032). First written 2026-09-18 for a single-seed
version of this repository; rewritten in November 2026 for the multi-seed results.

## Original paper claim

Fixed-size-state SSMs are fundamentally capped at copying: attention can re-read any prompt
position (a two-layer Transformer can in principle copy exponentially long strings), while
an SSM must compress the history into its state. Synthetic experiments (strings over a
26-letter alphabet, `$s|answer.` format) and pretrained-LLM evaluations support the claim.

## Our setup

- Task `ExactCopyDataset` (src/ssmbench/data/synthetic.py), following the paper's
  formulation: `<bos> <string> <sep>` prompt, emit `<string> <eos>`; loss only on the
  answer; 26-letter alphabet.
- Two training regimes: every string of length 32 (`configs/exact_copy.yaml`), or lengths
  drawn uniformly from 1 to 32 (`configs/exact_copy_varlen.yaml`). 8,000 strings, 6 epochs,
  batch 64, cosine schedule, three run seeds, deterministic training.
- The same trained model is evaluated at string lengths 16, 32, 64 and 128.
- Six variants, parameter-matched to the Transformer within 10%: Transformer with
  sinusoidal positions and with RoPE (602,782 parameters), S4D (604,102), Mamba-1 (597,102),
  Mamba-2 (548,690), Mamba-3 (550,625).
- Differences from the paper: much smaller models, our plain-PyTorch implementations, no
  ALiBi, unpacked single-example contexts. Classification: **qualitative reproduction**.

## Results table (task-dependent behavior; not a universal ranking)

Token accuracy in percent, mean over three seeds (five for parity); `results/SUMMARY.md`
has the standard deviations.

| Experiment | Train | Test | Transformer (best of two) | Best SSM | Interpretation |
|---|---|---|---|---|---|
| selective copy | T=64, 3 to 5 pairs | in distribution | 100.0 | 99.3 (Mamba-1) | any model that can select solves it; LTI S4D reaches 28.4 |
| selective copy | T=64, 3 to 5 pairs | T=128, 5 to 10 pairs | 12.1 | 52.1 (Mamba-1) | real selective SSMs keep about half the answers with a wide seed spread; Mamba-3 14.4 |
| exact copy | length 32 | 32 | 100.0 | 100.0 (S4D, Mamba-3) | a fixed delay; the LTI S4D represents it exactly |
| exact copy | lengths 1 to 32 | 32 | 98.6 (RoPE) | 61.4 (Mamba-3) | the paper's ranking: attention copies variable-length strings best |
| exact copy | either | 64, 128 | at most 6.4 | at most 13.1 | nobody extrapolates beyond the training lengths |
| parity | T=128 | T=128 | 52.5 | 72.6 (Mamba-3) | input-dependent rotation; two of five Mamba-3 seeds learn it, none with theta = 0 |

## Findings

1. Copying at a single training length is a fixed delay that the LTI S4D and Mamba-3 solve
   as well as attention, so it says nothing about state capacity.
2. Copying over variable lengths reproduces the paper's ranking at small scale: the
   Transformers lead (98.6 and 87.4% at length 32), Mamba-3 is the best SSM (61.4%), and
   S4D falls to 19.7%.
3. Mamba-1 and Mamba-2 learn exact copying slowly under the shared budget (their validation
   loss is still falling after six epochs), so their scores show slow learning rather than a
   capacity limit.
4. No model, Transformer included, extrapolates beyond its training lengths with sinusoidal
   or rotary positions; the paper's ALiBi result is not tested here.
5. With selective copying, the ranking **differs with task formulation**, which is the core
   research question of this extension.

## Failure modes and honesty

- One shared training budget for all models (no per-model tuning), mirroring the paper's
  fixed protocol.
- Three seeds per configuration; out-of-distribution selective copying and Mamba-3 parity
  have large spreads across seeds.
- No claim of exact reproduction: models, scale, encodings and packing differ from the paper.
- Nothing was tuned to force agreement with the paper.

## Conclusion

The Transformer-vs-SSM copying literature and our selective-copy benchmark are not in
conflict: they measure different bottlenecks (state capacity vs selectivity), and the
variable-length exact-copy runs now show the capacity side at small scale. The repository
treats every architecture-vs-architecture result as task-dependent, with five task regimes
measured (selective copy, exact copy at a fixed and at variable lengths, parity, IMDb).

Reproduction artifacts: configs/exact_copy.yaml, configs/exact_copy_varlen.yaml,
results/train/exact_copy/, results/train/exact_copy_varlen/, figures/fig_exact_copy.png,
docs/copy_task_comparison.md, docs/literature/repeat_after_me_reproduction.md,
docs/literature/interview_defense.md.
