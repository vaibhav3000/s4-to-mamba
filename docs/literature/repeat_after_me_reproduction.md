# Repeat After Me: Qualitative Reproduction and Task-Formulation Study

Paper: "Repeat After Me: Transformers are Better than State Space Models at
Copying" - Samy Jelassi, David Brandfonbrener, Sham M. Kakade, Eran Malach,
ICML 2024. arXiv:2402.01032. Official code: sjelassi/transformers_ssm_copy
(MIT license; consulted for the task formulation, no code copied).

## Research question

Not "can we re-run their experiments?" but: **how sensitive are
Transformer-vs-SSM conclusions on long-range memory to the exact copying task
formulation?** Our flagship already had a copying benchmark (selective
copying) where selective SSMs beat Transformers. This paper reports the
opposite ranking under a different formulation. Both cannot be universal;
investigating why is the extension.

## The original claim

A fixed-size-state SSM must compress the whole history into its state, so the
amount of copyable information is bounded by state size; a two-layer
Transformer can in principle copy exponentially long strings because attention
re-reads any prompt position at answer time. Their synthetic task:
`$<string>|<answer>.` over a 26-letter alphabet, loss only on the answer;
their models: Transformers with RoPE/NoPE/ALiBi/Hard-ALiBi (hidden 1024, 12
layers), official Mamba (state 32), LSTM.

## Our setup (and every difference that matters)

| Aspect | Paper | Us |
|---|---|---|
| Task format | `$s|answer.` | identical structure: `<bos> s <sep>` -> emit `s <eos>` |
| Alphabet | 26 letters | 26 letters |
| Train lengths | 5..20..50 (per run) | 5..32 |
| Eval lengths | up to 100+ | 32 (in-dist), 64, 128 (extrapolation) |
| Models | TF (RoPE/NoPE/ALiBi), official Mamba, LSTM | our educational TF (sinusoidal PE), S4D, Mamba-1, Mamba-2 |
| Scale | hidden 1024, 12 layers | d_model 128, 3 layers (604k / 364k / 358k / 96k params) |
| Training | 2000 steps, batch 8, packed contexts | 6 epochs x 125 batches, batch 64, one example per sequence |
| Context handling | packed multi-example 220-token contexts | single example, fixed 264-token window at all lengths |
| Hardware | unspecified GPU | RTX 4050 Laptop 6GB |
| Metric | string (exact-match) accuracy | full-sequence success + token accuracy |

Verdict: this is a **qualitative reproduction at reduced scale**, not an exact
one. The task formulation and metric are faithful; the models, scale, context
packing and positional encodings differ materially.

## Measured results (results/exact_copy_*.json, seed 42)

Full-sequence success / token accuracy at each string length:

| Model | L=32 (in-dist) | L=64 (extrap) | L=128 (extrap) |
|---|---|---|---|
| transformer | **1.000** / 1.000 | 0.000 / 0.036 | 0.000 / 0.032 |
| s4d | **1.000** / 1.000 | 0.000 / 0.042 | 0.000 / 0.032 |
| mamba2 | 0.000 / 0.150 | 0.000 / 0.060 | 0.000 / 0.025 |
| mamba | 0.000 / 0.106 | 0.000 / 0.038 | 0.000 / 0.017 |

Chance token accuracy is 1/30 = 0.033.

## Findings

1. **The paper's central ranking reproduces in-distribution.** At matched
   budget (identical data, optimizer, schedule, parameter scale), the
   Transformer solves exact copying perfectly at L=32 while the selective SSMs
   (Mamba-1, Mamba-2) remain far below at 0.11-0.15 token accuracy after the
   same 48k example presentations. Qualitatively consistent with Jelassi et
   al., who observe the same gap at 10x the scale.
2. **A nuance the paper's framing hides: gated-LTI S4D also solves in-dist
   copying.** S4D is LTI in its convolution, but our S4D block has an
   input-dependent gate, and the LTI kernel acts as a learnable bank of echo
   responses: within the trained length range (5..32) it can echo any prompt.
   It then collapses to chance the moment strings exceed the training range,
   exactly as a fixed-capacity mechanism should. "LTI cannot copy" is true for
   length generalization, not automatically for bounded in-distribution
   copying.
3. **Extrapolation failure is positional-encoding-dependent, and ours fails
   the way the paper says absolute PEs fail.** Our Transformer uses sinusoidal
   absolute positions; trained only on strings <= 32, it collapses to chance
   at L=64/128. Jelassi et al. found the same for NoPE-style models, while
   their ALiBi/RoPE variants extrapolated. So our extrapolation arm reproduces
   the paper's PE-sensitivity finding rather than their best case; we did not
   add ALiBi (documented future work).
4. **The ranking flips against our own selective-copy benchmark.** Same
   models, different formulation: on selective copying (test L=128) Mamba-2
   reaches 0.183 full-sequence success vs 0.0 for Transformer and S4D. On
   exact copying the Transformer is at 1.0 where the Mambas are at 0.0. The
   mechanism: selective copy requires keeping only k values (constant in
   length) and filtering noise (selectivity superpower); exact copy requires
   keeping L log2(V) bits (linear in length, capacity-bound) with no noise to
   filter.

## What our reproduction does NOT establish

- Not an exact reproduction: different models, scale, encodings, packing.
- Not evidence that Mamba "cannot copy" in general: at larger scale/training
  the official Mamba copies short strings well; our claim is the matched-budget
  in-distribution gap and the capacity-bound extrapolation collapse.
- Not a claim that either architecture universally dominates: the two copy
  tasks flip the ranking, which is the point.
- Single seed per run; no error bars.

## Failure analysis

- Mamba-1/2 learn slowly here (0.15 token accuracy after 48k examples, still
  rising at the end of the schedule). We kept one shared budget for all models
  rather than tuning per model, mirroring the paper's fixed protocol; a
  tuned/longer SSM run might narrow the in-dist gap but not the capacity bound.
- The Transformer extrapolation collapse is a positional-encoding artifact, an
  instructive failure mode: architecture and encoding choices both matter when
  claiming "Transformers extrapolate".

## References

- Jelassi, Brandfonbrener, Kakade, Malach. Repeat After Me. ICML 2024.
  arXiv:2402.01032. Code: sjelassi/transformers_ssm_copy (MIT).
- Gu, Dao. Mamba. arXiv:2312.00752 (selective copying task origin).
- Our own selective-copy results: results/selective_copy_*.json.
