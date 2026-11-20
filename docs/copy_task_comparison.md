# Copy Task Comparison: Selective Copying vs Exact Copying

Two benchmarks in this repository both carry the word "copy", yet they make opposite demands
on a sequence model and produce different rankings. This document pins down the differences
so the result sets can be compared honestly. Numbers are means over three run seeds with
parameter-matched models (`results/SUMMARY.md`).

## Side by side

| Aspect | Selective Copying (marker-value recall after the Mamba paper's task) | Exact Copying (Repeat After Me style, Jelassi et al. 2024) |
|---|---|---|
| Prompt | noise stream with k marker->value pairs embedded, then a separator | `<bos> <string> <sep>` |
| Target | the k remembered values, in order | the full string, token for token, then `<eos>` |
| Information to retain | k values only: about k log2(V) bits, constant in sequence length | the entire string: L log2(V) bits, linear in length |
| Noise handling | essential skill: LTI models cannot gate noise out of state | none: there is no noise |
| Training | L=64 with 3 to 5 pairs | strings of length 32 only (`exact_copy`), or lengths 1 to 32 (`exact_copy_varlen`) |
| Testing | in distribution, and out of distribution at L=128 with 5 to 10 pairs | string lengths 16, 32, 64, 128 |
| Metric | token accuracy + full-sequence success | token accuracy + full-sequence success |
| What it stresses | selectivity: what to store vs forget | state capacity: how much can be stored |
| Measured (token accuracy) | in distribution: Transformers 100%, Mamba-1/2/3 98.3 to 99.3%, S4D 28.4%. Out of distribution: Mamba-1 52.1%, Mamba-2 49.6%, S4D 17.6%, Mamba-3 14.4%, Transformers 7.5 to 12.1% | fixed length 32: Transformers, S4D and Mamba-3 100%, Mamba-1/2 21.8 and 23.5%. Lengths 1 to 32, tested at 32: Transformer with RoPE 98.6%, sinusoidal 87.4%, Mamba-3 61.4%, Mamba-1 35.6%, Mamba-2 29.0%, S4D 19.7%. Every model at most 13.1% at 64 and 128 |

## Why the conclusions differ

A fixed-size state SSM must decide what to keep. Under selective copying that decision is
easy and the required state is tiny (a handful of values), so the selective gate is what
matters and the noise-filtering requirement defeats LTI models. Under exact copying every
token matters, so the state must hold L log2(V) bits and its fixed size becomes the binding
constraint. A Transformer never compresses: attention re-reads any prompt position at answer
time, so its effective memory grows with context. This is the mechanism argued in Jelassi et
al. (ICML 2024), "Repeat After Me".

The fixed-length exact-copy runs do not test that mechanism: with one training length, the
answer is the prompt shifted by a fixed offset, a fixed delay that the LTI S4D represents
exactly. The variable-length runs do: once the delay changes from example to example, S4D
falls to 19.7% at length 32, every SSM falls behind the Transformers, and Mamba-3 is the best
of them. Nothing generalizes beyond the longest training length, the Transformers included.

## The takeaway recorded in this repository

No single copy task settles the Transformer-vs-SSM question. Selective copying in
distribution favors any model that can select (attention or a selective SSM); its
out-of-distribution test favors Mamba-1 and Mamba-2, with a wide spread across seeds.
Variable-length exact copying favors attention, as the paper reports. Treat all such results
as task-dependent; see docs/literature/repeat_after_me_reproduction.md for the reproduction
and docs/literature/interview_defense.md for the interview framing.
