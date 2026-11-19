# ssmbench

A study of efficient sequence models (S4D, Mamba, Mamba-2, Mamba-3) against a Transformer
baseline: what each one can learn, and what it costs to train and to generate with.

## Models

**Architecture**:
One of the five sequence-model families in the study: Transformer, S4D, Mamba (Mamba-1),
Mamba-2, Mamba-3.
_Avoid_: backbone (when the family is meant), model type

**Model variant**:
An architecture with a fixed set of settings, named by a label such as `transformer_rope` or
`mamba3_no_rotation`; the unit that results are reported for.
_Avoid_: model (when a specific setting is meant), config

**Mixer**:
The part of a block that moves information across positions: attention in the Transformer, the
state space layer in the SSMs.
_Avoid_: layer, core

**State space model (SSM)**:
A mixer that carries a fixed-size state through time, updated by a linear recurrence.

**Selective**:
Said of an SSM whose transition and input weights depend on the current token (Mamba-1, 2, 3);
S4D is not selective (it is linear time-invariant).
_Avoid_: input-dependent (as a label), gated

**Rotation**:
The imaginary part of Mamba-3's state transition, which lets the state turn as well as decay.
**Rotation ablation**: the Mamba-3 variant with rotation removed, so its transition only decays.

**Exponential-trapezoidal rule**:
Mamba-3's discretization, which weights the input at both ends of each step by lambda and
1 - lambda; lambda = 1 is the exponential-Euler rule of Mamba-1 and Mamba-2.

**Two-band correction**:
The extra input weight that the exponential-trapezoidal rule puts on the previous position; in
the parallel form it adds one band below the diagonal of the input weights.

**Boundary input**:
The previous position's input that the exponential-trapezoidal rule still owes to the next
step; it crosses chunk boundaries and decode steps together with the state.

**Positional encoding**:
How the Transformer learns token order: an additive sinusoidal table, or RoPE (rotary position
embedding of queries and keys). SSMs have none.

**Attention backend**:
The kernel family that computes attention: FlashAttention, memory-efficient, cuDNN, math, or the
naive reference.

## Forms of one model

**Parallel form**:
The computation over a whole sequence at once: FFT convolution (S4D), chunked SSD (Mamba-2,
Mamba-3), causal attention (Transformer).
_Avoid_: training mode, batch form

**Recurrent form**:
The same model computed one position at a time from a carried state.
_Avoid_: RNN mode, inference mode

**Chunked form**:
A parallel form that splits the sequence into chunks, computes each chunk with matrix
products, and passes only a small state between chunks.

**Reference**:
A slow, plainly correct implementation used only as the oracle that faster forms are tested
against.
_Avoid_: baseline (a baseline is a model the study compares against)

## Generation

**Prompt**:
The tokens given before generation starts.

**Prefill**:
Processing the prompt in the parallel form and producing the decode state.

**Decode step**:
Producing one more token's output from the decode state, and updating that state.
_Avoid_: inference step, forward step

**Decode state**:
Everything a model carries between decode steps: the KV cache for the Transformer, the
recurrent state (plus convolution history and boundary input) for an SSM.
_Avoid_: cache (for SSMs), hidden

**Time to first token (TTFT)**:
The time from the prompt being on the device to the first generated token being known.

## Experiments

**Task**:
A learning problem with a fixed data distribution and a stated chance level: parity, selective
copy, exact copy, IMDb sentiment.

**Run**:
One training of one model variant on one task with one run seed.

**Run seed**:
The seed that sets initialization, dropout and data order of a run; the datasets do not depend
on it.
_Avoid_: data seed (a different seed that fixes the examples)

**Evaluation set**:
A fixed, named set of examples a trained model is scored on, such as `T=512` or `ood`.

**Validation set**:
The evaluation set from the training distribution used to follow training; never used to report
results.

**Test evaluation set**:
An evaluation set scored once after training; reported results come only from these.

**In distribution / out of distribution**:
Whether an evaluation set comes from the training distribution (lengths, pair counts) or not.
_Avoid_: ID/OOD in prose

**Length extrapolation**:
Scoring a model on sequences longer than any it was trained on.

**Summary**:
The mean, sample standard deviation and count over the runs of one variant on one evaluation
set.

**Chance level**:
The score a model gets by guessing from the label distribution alone.

**Parameter-matched**:
Said of variants whose total parameter counts are within a stated tolerance of each other.
_Avoid_: same-size, fair (unqualified)

**Width-matched**:
Said of variants with the same model width and depth, whatever their parameter counts.

## Measurement

**Sweep point**:
One (variant, precision, sequence length) measured by a benchmark.

**Precision**:
The numeric format a measurement runs in: fp32, or bf16/fp16 autocast.

**Peak allocated / peak reserved memory**:
The most GPU memory held by tensors / by the allocator's cache during a measured phase.

**Spill**:
A phase whose memory ran past what the GPU had free, so the driver paged to system memory and
its timing is not a GPU-memory number.

**Provenance**:
The record of which code, environment and settings produced a result.
