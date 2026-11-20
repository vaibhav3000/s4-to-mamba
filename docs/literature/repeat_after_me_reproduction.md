# Repeat After Me: Qualitative Reproduction and Task-Formulation Study

Paper: "Repeat After Me: Transformers are Better than State Space Models at
Copying", Samy Jelassi, David Brandfonbrener, Sham M. Kakade, Eran Malach,
ICML 2024. arXiv:2402.01032. Official code: sjelassi/transformers_ssm_copy
(MIT license; consulted for the task formulation, no code copied).

## Research question

Not "can we re-run their experiments?" but: **how sensitive are Transformer-vs-SSM
conclusions on long-range memory to the exact copying task formulation?** This repository
also has a selective-copying benchmark, on which selective SSMs do well. This paper reports a
ranking in favor of Transformers under a different formulation. Investigating why is the
extension.

## The original claim

A fixed-size-state SSM must compress the whole history into its state, so the amount of
copyable information is bounded by state size; a two-layer Transformer can in principle copy
exponentially long strings because attention re-reads any prompt position at answer time.
Their synthetic task: `$<string>|<answer>.` over a 26-letter alphabet, loss only on the
answer; their models: Transformers with RoPE/NoPE/ALiBi/Hard-ALiBi (hidden 1024, 12 layers),
official Mamba (state 32), LSTM.

## Our setup (and every difference that matters)

| Aspect | Paper | Us |
|---|---|---|
| Task format | `$s|answer.` | identical structure: `<bos> s <sep>` -> emit `s <eos>` |
| Alphabet | 26 letters | 26 letters |
| Train lengths | varied per run | two configs: length 32 only (`configs/exact_copy.yaml`) and lengths drawn uniformly from 1 to 32 (`configs/exact_copy_varlen.yaml`) |
| Eval lengths | up to 100+ | 16, 32, 64, 128 |
| Models | TF (RoPE/NoPE/ALiBi), official Mamba, LSTM | our Transformer with sinusoidal positions and with RoPE, S4D, Mamba-1, Mamba-2, Mamba-3 |
| Scale | hidden 1024, 12 layers | 3 layers, parameter-matched to the Transformer within 10% (548,690 to 604,102 parameters) |
| Training | 2000 steps, batch 8, packed contexts | 6 epochs over 8,000 strings, batch 64, one example per sequence, three run seeds |
| Hardware | unspecified GPU | RTX 4050 Laptop 6 GB |
| Metric | string (exact-match) accuracy | full-sequence success + token accuracy |

Verdict: a **qualitative reproduction at reduced scale**, not an exact one. The task format
and metric follow the paper; the models, scale, context packing and positional encodings
differ.

## Measured results (results/train/exact_copy*/, three seeds)

Token accuracy in percent at each string length (mean over seeds; the README has the
standard deviations). Chance is 1/26 = 3.8% per letter.

Trained at length 32 only:

| Model | L=16 | L=32 (trained) | L=64 | L=128 |
|---|---|---|---|---|
| Transformer (sinusoidal) | 8.7 | 100.0 | 3.6 | 3.6 |
| Transformer (RoPE) | 3.4 | 100.0 | 3.9 | 3.5 |
| S4D | 3.0 | 100.0 | 4.6 | 2.7 |
| Mamba-1 | 10.1 | 21.8 | 5.1 | 1.6 |
| Mamba-2 | 16.8 | 23.5 | 7.1 | 3.0 |
| Mamba-3 | 2.5 | 100.0 | 3.6 | 2.8 |

Trained on lengths 1 to 32:

| Model | L=16 | L=32 | L=64 | L=128 |
|---|---|---|---|---|
| Transformer (sinusoidal) | 100.0 | 87.4 | 5.0 | 4.0 |
| Transformer (RoPE) | 100.0 | 98.6 | 6.4 | 4.1 |
| S4D | 21.4 | 19.7 | 5.9 | 3.2 |
| Mamba-1 | 62.6 | 35.6 | 13.1 | 5.0 |
| Mamba-2 | 50.0 | 29.0 | 12.4 | 4.8 |
| Mamba-3 | 80.2 | 61.4 | 5.8 | 3.8 |

## Findings

1. **At a single training length, copying is a fixed delay.** The Transformers, S4D and
   Mamba-3 copy perfectly at length 32 and fail at every other length. S4D's success is the
   tell: an LTI convolution represents a shift by a constant offset exactly, so this
   configuration needs no selectivity and says nothing about state capacity.
2. **Variable lengths reproduce the paper's ranking.** Once the delay varies, the
   Transformers copy best (98.6% at length 32 with RoPE, 87.4% with sinusoidal positions),
   Mamba-3 is the best SSM (61.4%), Mamba-1 and Mamba-2 reach 35.6 and 29.0%, and S4D falls
   to 19.7%. This is the gap Jelassi et al. report, here at much smaller scale.
3. **Mamba-1 and Mamba-2 learn slowly at this budget.** Trained at length 32, their
   validation loss falls every epoch and is still falling after the sixth (for example from
   3.20 to 2.06 for Mamba-1 seed 0, against ln 26 = 3.26 for uniform guessing), while Mamba-3
   and the sinusoidal Transformer end between 0.001 and 0.003. Their low scores are slow learning under a shared
   budget, not a demonstrated capacity limit.
4. **Nobody extrapolates.** Every model, both Transformers included, is at most 13.1% at
   lengths 64 and 128. Jelassi et al. report that Transformer length generalization on this
   task depends on the positional encoding (ALiBi extrapolates); neither of our encodings
   does at this scale, and we did not train ALiBi.
5. **The ranking differs from selective copying.** On selective copying in distribution,
   the selective SSMs and the Transformers all solve the task and S4D does not; out of
   distribution, Mamba-1 and Mamba-2 keep about half the answers while the Transformers and
   Mamba-3 collapse. Exact copying over variable lengths favors attention. Same
   architectures, different bottleneck.

## What our reproduction does NOT establish

- Not an exact reproduction: different models, scale, encodings and packing.
- Not evidence that Mamba "cannot copy" in general: larger models trained longer copy short
  strings well, and our Mamba-1/2 runs were still improving.
- Not a test of ALiBi or of the paper's pretrained-model experiments.
- Three seeds per configuration; the per-seed spread is in the README tables.

## References

- Jelassi, Brandfonbrener, Kakade, Malach. Repeat After Me. ICML 2024. arXiv:2402.01032.
  Code: sjelassi/transformers_ssm_copy (MIT).
- Gu, Dao. Mamba. arXiv:2312.00752 (selective copying task origin).
- Our results: results/train/exact_copy/, results/train/exact_copy_varlen/,
  results/train/selective_copy/.
