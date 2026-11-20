# Copy Task Comparison: Selective Copying vs Exact Copying

> **Status (November 2026).** This note was written for the earlier single-seed version
> of this repository, whose models had defects that have since been fixed. The numbers and
> result files it quotes are kept for the record (the files are in the git history before
> commit `ab29379`) and are superseded by the three-seed results in `results/SUMMARY.md`,
> the README and the report.

Two benchmarks in this repository both carry the word "copy", yet they make
opposite demands on a sequence model and, in our runs, produce different rankings. This
document pins down the differences precisely so the two result sets can be
compared honestly. The numbers come from single-seed runs that predate later fixes to the
models (see `results/PROVENANCE.md`), so they are stale until re-run.

## Side by side

| Aspect | Selective Copying (ours; marker-value recall after the Mamba paper's task) | Exact Copying (Repeat After Me style, Jelassi et al. 2024) |
|---|---|---|
| Prompt | noise stream with k marker->value pairs embedded, then a separator | `<bos> <string> <sep>` |
| Target | the k remembered values, in order (k in 3..5 in training, 5..10 at test) | the full string, token for token, then `<eos>` |
| Information to retain | k values only: ~k log2(V) bits, CONSTANT in sequence length | the entire string: L log2(V) bits, LINEAR in length |
| Noise handling | essential skill: LTI models cannot gate noise out of state | none: there is no noise |
| Train lengths | L=64 (3-5 pairs) | strings of length 32 only (every training string has exactly 32 letters) |
| Test lengths | L=128 (5-10 pairs) | 32 (in-distribution), 64, 128 (extrapolation) |
| Metric | token accuracy + full-sequence success | token accuracy + full-sequence success |
| What it stresses | selectivity: what to store vs forget | state capacity: how much can be stored |
| Our measured result (one seed, pre-fix code) | Mamba-2 and Mamba-1 score highest (mamba2 0.670 / mamba 0.649 token accuracy; s4d 0.064; transformer 0.205), measured out of distribution in both length and pair count | Transformer and S4D both reach 1.000 sequence success at L=32, Mamba-1/2 0.000; every model is at 0.000 at L=64 and 128 (results/exact_copy_*.json) |

## Why the conclusions differ

A fixed-size state SSM must decide what to keep. Under selective copying that
decision is easy and the required state is tiny (a handful of values), so the
selective gate is a superpower and the noise-filtering requirement destroys
LTI models. Under exact copying the decision is impossible: every token
matters, so the state must hold L log2(V) bits and the fixed size becomes the
binding constraint. A Transformer never compresses: attention re-reads any
prompt position at answer time, so its effective memory grows with context.

This is exactly the mechanism argued in Jelassi et al. (ICML 2024),
"Repeat After Me": a two-layer attention model can in principle copy strings
of exponential length, while a fixed-state model is capped by its state.

Our exact-copy runs do not demonstrate this mechanism. Training used one string length (32),
so the answer is always the prompt shifted by a fixed offset, a fixed delay that the LTI S4D
also learned perfectly. The Mamba-1/2 validation loss was still falling when the 6-epoch
schedule ended, and their states hold far more numbers than a 32-letter string needs. Their
0.000 is better read as slow learning under this budget than as a capacity limit.

## The takeaway recorded in this repository

No single copy task settles the Transformer-vs-SSM question. The two
formulations are designed to probe different axes (selectivity vs state capacity), and the
ranking differs between them in our runs. The comparison is uneven, though: the selective-copy
score is out of distribution (longer sequences and more pairs than in training), while the
exact-copy score that separates the models is in distribution at the single training length,
where S4D matches the Transformer. Treat all such results as task-dependent; see
docs/literature/repeat_after_me_reproduction.md for our measured reproduction
and docs/literature/interview_defense.md for the interview framing.
