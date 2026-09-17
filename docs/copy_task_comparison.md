# Copy Task Comparison: Selective Copying vs Exact Copying

Two benchmarks in this repository both carry the word "copy", yet they make
opposite demands on a sequence model and produce opposite rankings. This
document pins down the differences precisely so the two result sets can be
compared honestly.

## Side by side

| Aspect | Selective Copying (ours, Mamba-paper style) | Exact Copying (Repeat After Me style, Jelassi et al. 2024) |
|---|---|---|
| Prompt | noise stream with k marker->value pairs embedded, then a separator | `<bos> <string> <sep>` |
| Target | the k remembered values, in order (k in 3..5) | the full string, token for token, then `<eos>` |
| Information to retain | k values only: ~k log2(V) bits, CONSTANT in sequence length | the entire string: L log2(V) bits, LINEAR in length |
| Noise handling | essential skill: LTI models cannot gate noise out of state | none: there is no noise |
| Train lengths | L=64 (3-5 pairs) | strings of length 5..32 |
| Test lengths | L=128 (5-10 pairs) | 32 (in-distribution), 64, 128 (extrapolation) |
| Metric | token accuracy + full-sequence success | token accuracy + full-sequence success |
| What it stresses | selectivity: what to store vs forget | state capacity: how much can be stored |
| Our measured winner | selective SSMs (mamba2 0.670 / mamba 0.649 token accuracy; s4d 0.064; transformer 0.205) | Transformer (see results/exact_copy_*.json for the measured numbers) |

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

## The takeaway recorded in this repository

No single copy task settles the Transformer-vs-SSM question. The two
formulations probe two different axes (selectivity vs state capacity), and the
ranking flips between them. Treat all such results as task-dependent; see
docs/literature/repeat_after_me_reproduction.md for our measured reproduction
and docs/literature/interview_defense.md for the interview framing.
