# Interview guide: state space models from S4 to Mamba-3

Questions and answers tied to the actual code in `src/ssmbench/models/`, the tests in
`tests/test_models.py`, and the committed results in `results/`. The answers are written to be
said out loud in an interview: exact where the mathematics is exact, honest where the
implementation is educational. See `docs/evolution.md` for the full narrative and
`REFERENCES.md` for sources.

## A. The core abstraction

### 1. What is "the state" in these models?

The state `h` is a fixed-size tensor that summarizes everything the model keeps about the
prefix. Its shape is the memory budget. In our files: S4D carries a complex `(B, D, N)` state
(`S4DLayer.step`), Mamba-1 carries `(B, d_inner, N)` plus the last `d_conv - 1` conv inputs
(`MambaBlock.init_state`), Mamba-2 carries an `(N, P)` matrix per head, Mamba-3 a complex
`(N, P)` matrix per head. Between tokens, the state is the only memory; there is no growing
cache as in a Transformer's KV cache.

### 2. How does the recurrence work, concretely, in your code?

`MambaBlock.selective_scan` in `src/ssmbench/models/mamba_minimal.py` is the plainest version:

```
alpha = exp(delta_t * A)                     # A is a real negative diagonal (d_inner, N)
h     = alpha * h + delta_t * B_t * x_t
y_t   = C_t . h + D * x_t
```

One python loop over `t`. `alpha` is the fraction of the state that survives the step, the
`delta_t * B_t * x_t` term is what the current token writes into it, and `C_t` reads the state
out. The same pattern with different `A` and different discretization weights appears in
`Mamba2Block.sequential_scan` and `Mamba3Block.selective_scan`.

### 3. Why do SSM layers need no positional embeddings while the Transformer does?

The recurrence is order-sensitive by construction: `h_t` is computed from `h_{t-1}`, so
shuffling the input changes every state. That is why `S4DBackbone` and the Mamba backbones
have no positional table. Attention is permutation-equivariant without one, so
`TransformerBackbone` adds a sinusoidal table (`sinusoidal_positions` in `blocks.py`) once
before the blocks.

### 4. What does delta do, and why is it passed through a softplus?

`delta` is the step size, the per-token timescale. It controls how much of the state is kept
versus overwritten: with `alpha = exp(delta * A)` and `A < 0`, small `delta` keeps `alpha`
near 1 (long memory), large `delta` pushes `alpha` toward 0 (forget and write). The softplus
guarantees positivity, which keeps `alpha` in `(0, 1)` and the system stable. Initialization
matters: `dt_proj` bias is set so `softplus(output)` starts log-uniform in `[1e-3, 1e-1]`
(`_init_dt` in both `mamba_minimal.py` and `mamba2_minimal.py`), the same convention as the
official implementation.

### 5. Why ZOH instead of plain Euler?

Zero-order hold matches how sequences are consumed: the input is constant during one step, and
ZOH is the exact solution of the ODE under that assumption. It gives `Abar = exp(delta A)`,
which keeps the transition contractive whenever `Re(A) < 0`. Explicit Euler approximates the
transition itself and is only first-order; on a stable continuous system a too-large step can
push its eigenvalues outside the unit circle, which `exp` can never do. The input-side ZOH
integral contributes the exact factor `(e^z - 1)/z` with `z = delta * A`, computed in
`zoh_discretization` in `s4_minimal.py` with a Taylor fallback near `z = 0`.

## B. S4 and S4D

### 6. What is the difference between S4 and S4D, and which do you implement?

S4 uses a dense structured transition matrix (Normal-Plus-Low-Rank) initialized from HiPPO,
and needs the Woodbury identity to keep its convolution kernel computable. S4D restricts `A`
to complex diagonal, which collapses the kernel to `K_k = sum_n C_n Bbar_n Abar_n^k`, a sum
over state channels, with almost no quality loss. We implement S4D only
(`src/ssmbench/models/s4_minimal.py`), with S4D-Lin initialization `A_n = -1/2 + i * pi * n`.
Full S4 NPLR plus Woodbury is referenced, not reimplemented, and the module docstring says so.

### 7. Why does the convolution form work for S4 but not for Mamba?

Because S4 is LTI: `A, B, C, delta` are constants, so the discrete map from the input prefix
to the output is one fixed causal convolution with kernel `K_k = C Bbar Abar^k`, computable by
FFT in O(T log T). In Mamba, `delta_t, B_t, C_t` depend on the token, so the system is
time-varying; the effective kernel would depend on the whole input sequence, so there is
nothing fixed to convolve with. That is why `S4DLayer.forward` is an FFT convolution and
`MambaBlock.forward` is a scan.

### 8. What is HiPPO initialization actually buying you?

A principled answer to "what should a fixed-size memory contain". The HiPPO matrix defines a
linear map whose state holds the coefficients of an orthogonal polynomial projection of the
past, meaning the state is the best low-order summary of the history in a precise sense. S4
and S4D inherit this as their initialization: the imaginary parts `pi * n` in S4D-Lin place
each state channel at a different rotation frequency, so different channels remember at
different timescales. It is an initialization and a parameterization, not a training-time
constraint; everything is learnable afterwards.

### 9. How do you know your S4D training form and inference form are the same model?

`tests/test_models.py::test_s4d_conv_equals_recurrent` runs the FFT-convolution forward and
then `T` calls of `S4DLayer.step` on the same input and requires max error below 1e-3. The two
paths are derived from the same discretized parameters, so they must agree to floating-point
tolerance; a failure would mean the kernel constants or the ZOH factor are inconsistent with
the recurrence.

### 10. What is the `(e^z - 1)/z` factor and what goes wrong without it?

It is the exact ZOH input integral: `Bbar = A^{-1}(exp(delta A) - 1) B = delta B (e^z - 1)/z`.
Dropping it and using `delta B` is a first-order approximation of the input contribution.
For S4D that would make the kernel inconsistent with the recurrent `step()`, and our
equivalence test would fail, because the training path and the decode path would no longer be
the same system. It also matters at large `delta * A`, where the ratio is far from 1.

## C. Mamba-1: selectivity

### 11. What exactly does "selective" mean?

That the mixing parameters are functions of the input token. In `mamba_minimal.py`,
`x_proj` maps each convolved token to `(delta_rank, B_t, C_t)` and `dt_proj` maps the rank
bottleneck to per-channel `delta_t`. So the amount of forgetting (`delta_t`), what is written
(`B_t`), and what is read (`C_t`) all change per token. In S4/S4D they are learned constants
shared by all tokens.

### 12. Why is Mamba not an LTI system, and why does that matter?

LTI means linear and time-invariant: the same operator acts at every step, which is exactly
what makes the FFT convolution possible (question 7). Mamba is linear but time-varying. The
practical consequences: no convolution form, so training runs a scan; and the model can
finally filter by content, keeping relevant tokens and ignoring noise, which no LTI model can
do regardless of how long it trains. Our selective-copying result shows the second point
empirically (question 29).

### 13. Why does Mamba still have a short convolution if the SSM already mixes across time?

The SSM mixes through the state, which compresses the whole prefix into a fixed-size vector.
The depthwise causal conv of width `d_conv = 4` restores cheap direct mixing between adjacent
tokens before they enter the SSM, which the official design keeps for exactly that reason.
It also gives the parameter projections (`delta, B, C`) a locally smoothed token to look at.

### 14. What is the role of the z gate?

`in_proj` produces two `d_inner` branches, a data branch and a gate branch `z`. After the SSM,
the block computes `out_proj(y * silu(z))`. This is a per-channel multiplicative gate on the
SSM output, letting the network suppress or pass the state-based feature per position and
channel. It is part of the Mamba block design and is replicated exactly in the incremental
`step()` path.

### 15. Is your Mamba actually O(T)?

In operations, yes: the scan touches each token once with constant work per token, and decode
is O(1) per token with a fixed-size state. In wall-clock, no, not in this repository: the
scan is a python loop, so each step launches small eager-mode kernels and the per-step launch
overhead dominates; asymptotic O(T) with a large constant is still slow. The official fused
selective-scan kernels exist precisely to remove that overhead. Our length sweep measures the
gap (`scripts/run_efficiency.py`, results in `results/efficiency_gpu.json`), and the per-epoch
`wall_time_seconds` in every `results/*.json` shows it too. Being able to separate operation
count from wall-clock is part of what this repository teaches.

### 16. How does incremental decoding work and what state does each block carry?

`S4DBlock.step` carries a complex `(B, D, N)` state and updates it with
`new_h = Abar * h + Bbar * x_t`, O(1) per token. `MambaBlock.step` carries `(B, d_inner, N)`
plus the last `d_conv - 1` conv inputs, so the short convolution can be computed from history
instead of re-reading the prefix. Both are verified: `test_mamba_step_equals_forward` shows
the step path reproduces the parallel forward to 1e-3. Mamba-2 and Mamba-3 have no implemented
decode path in this repo (their conceptual state is the `(N, P)` matrix per head), which we
state plainly in `docs/evolution.md`.

## D. Mamba-2 and SSD

### 17. What is State Space Duality?

The observation, formalized in the Mamba-2 paper, that a scalar-decayed SSM layer has two
equivalent algorithms: a recurrence over a small state, and an attention-like quadratic form.
Within a chunk of tokens the output is a masked, decay-weighted matmul with scores
`(C_t . B_s)`; across chunks, only the `(N, P)` state is carried. Same layer, two compute
shapes. Our `Mamba2Block` implements both (`chunked_scan` and `sequential_scan`) and the test
requires them to agree.

### 18. Why does Mamba-2 exist? What do tensor cores have to do with it?

GPUs execute large matmuls (tensor cores) far more efficiently than long chains of small
dependent operations. Mamba-1's scan is the latter. Mamba-2 restricts `A` to a scalar times
identity per head, which is exactly the restriction that makes the recurrence algebraically
unrollable into matmuls within a chunk. The chunk loop then runs `T / chunk_size` iterations
instead of `T`, each doing matmul-shaped work. The restriction trades a little expressive
power (scalar decay, no per-entry transition) for hardware efficiency, and the paper argues
the trade is favorable.

### 19. Write down the chunk equations.

For a chunk with `cs = cumsum(log alpha)` inside the chunk, carried state `S` of shape
`(N, P)`:

```
L_ts      = exp(cs_t - cs_s)   for t >= s
y_intra_t = sum_{s <= t} L_ts * (C_t . B_s) * x_s
y_inter_t = exp(cs_t) * (C_t . S)
S_new     = exp(cs_end) * S + sum_t exp(cs_end - cs_t) * B_t (outer) x_t
y_t       = y_intra_t + y_inter_t
```

These are the docstring equations of `Mamba2Block.chunked_scan`, implemented with einsums:
`L * attn` is the decayed score matrix, `y_intra` the intra-chunk matmul, `y_inter` the
carried-state read, and `S_partial` plus the chunk-end decay the state update.

### 20. How do you compute the decay matrix L without numerical problems?

In log space. `log_alpha` is cumsum'd inside the chunk, differences `cs_t - cs_s` are formed,
clamped to `max = 0` (lower-triangular entries are already non-positive; upper-triangular
ones can grow to the chunk length in magnitude and overflow fp32 `exp`), then exponentiated
and masked with a lower-triangular boolean. Padding handles lengths that are not multiples of
`chunk_size`. The clamping is correctness-relevant at long chunks, not a style choice.

### 21. Why are B and C shared across heads (`n_groups`)?

Parameter and compute economy: with `n_groups = 1`, one `B_t` and one `C_t` vector serve all
heads, so the `x_proj` output for `B` and `C` is much smaller and the score `(C_t . B_s)` is
computed once per chunk pair. Heads keep private transitions (`a_h`, `delta_{h,t}`) and
private values `x_{h,t}`. This is the Mamba-2 group-variant design; `n_groups > 1` buys
per-group read/write vectors back.

## E. Mamba-3

### 22. What changed in Mamba-3? Give all three.

From arXiv:2603.15569, as implemented in `mamba3_minimal.py`:

1. Exponential-trapezoidal discretization: the input integral uses both interval endpoints,
   `h_t = alpha_t h_{t-1} + beta_t B_{t-1} x_{t-1} + gamma_t B_t x_t` with
   `beta_t = (1 - lambda_t) delta_t alpha_t` and `gamma_t = lambda_t delta_t`, where
   `lambda_t` is predicted per token. Second-order accurate in delta; `lambda = 1` recovers
   Mamba-1/2.
2. Complex-valued transitions: `A = -exp(a_real) + i theta`, so `alpha_t` is contractive
   (`|alpha_t| < 1`) and rotating at the same time. This restores state-tracking ability that
   real non-negative transitions provably lack.
3. MIMO formulation: multiple input/output channels per state slot, improving quality without
   increasing decode latency. Described only; our implementation keeps SISO per head and the
   module docstring says so.

### 23. Justify the exponential-trapezoidal rule. Why does beta contain a factor of alpha?

The continuous update integrates `B(t) x(t)` against the decaying transition over one step.
The right-endpoint-only rule (exponential-Euler, Mamba-1/2) is first-order: it samples the
integrand only at the end of the interval. The trapezoidal rule samples both endpoints and
averages them with a data-dependent weight `lambda_t`, which cancels the leading error term,
making it second-order accurate in delta. The left-endpoint contribution must be multiplied by
`alpha_t = exp(delta_t A_t)` because that input has been held across the whole interval and
its effect decays by the transition; that is why
`beta_t = (1 - lambda_t) delta_t alpha_t` while `gamma_t = lambda_t delta_t` does not carry
the factor.

### 24. What does lambda do at lambda = 1, 1/2, and 0?

At `lambda = 1`: `beta = 0`, `gamma = delta`, which is exactly the Mamba-1/2
exponential-Euler update; `test_mamba3_reduces_to_exp_euler` pins this by zeroing `theta` and
forcing `lambda` to 1 and checks equality with a hand-written Mamba-1/2 reference to 1e-3. At
`lambda = 1/2`: the classical trapezoidal rule, which is also our zero-initialized starting
point (`lambda_proj` weights and bias start at zero, so `sigmoid(0) = 0.5`). At `lambda = 0`:
the pure left-endpoint rule, all input injected as `delta_t alpha_t B_{t-1} x_{t-1}` and
nothing at the right endpoint. `lambda_t` is data-dependent (a sigmoid gate on a per-head
projection), so the model can interpolate the rule per token and per head.

### 25. Why do complex transitions fix parity while real non-negative ones cannot, in one sentence?

Parity requires the hidden state to flip according to the input bit, which is a rotation-like
operation, and a real non-negative matrix can only scale (decay or grow) the state, never
rotate it; a complex phase `exp(i delta theta)` can. Our numbers: on cumulative XOR
(L = 128, 20 epochs), mamba3 reaches 0.9834 token accuracy with 0.42 sequence success while
mamba2, mamba, transformer, and s4d sit at 0.563, 0.559, 0.533, and 0.506 token accuracy with
0.0 sequence success (`results/parity_*.json`).

### 26. S4D also has complex A. Why does it still fail parity?

Because rotation alone is not enough; the transition must be input-dependent. S4D is LTI: its
`A, B, C, delta` are constants shared across time, so the state evolves as a fixed linear
system driven additively by the input. Parity needs the transition's effect to depend on the
current bit, which no fixed linear system represents. Mamba-3 has both properties at once:
selective (input-dependent `delta, B, C, lambda`) and rotating (complex `A`). Selectivity
without rotation fails (Mamba-1/2), rotation without selectivity fails (S4D), the
combination works (Mamba-3).

## F. Expressivity and benchmarks

### 27. Why does parity kill real-transition SSMs? What is the formal background?

With real non-negative eigenvalues the transition is a pure scaling: any product of such
transitions is again a scaling, so the state trajectory cannot implement the sign/phase
switches that parity (cumulative XOR) requires. The Mamba-3 paper's discussion of
expressivity covers this, drawing on results by Grazzi et al. on what state-tracking problems
recurrent and state space models can represent and by Merrill et al. on the limits of
real-valued transitions; we cite it through that discussion rather than restating the theory.
Empirically our benchmark agrees: all four real-transition baselines solve zero full parity
sequences and sit at token accuracy 0.506 to 0.563, near the 0.5 chance rate, while the
complex-transition model reaches 0.9834 with 0.42 full-sequence success.

### 28. What breaks at 10x context, for each model?

Qualitatively, from the compute shapes: the Transformer's attention score compute is O(T^2),
so 10x context is 100x attention work, and the KV cache grows linearly with T; SSM states
stay constant. S4D's training form recomputes and FFT-convolves with an O(T log T) kernel and
O(T) activation memory. Mamba-1 and Mamba-3 scan with a python loop, so wall-clock grows
linearly in T with a large per-step constant. Mamba-2's chunk loop is `T / chunk_size`
iterations with quadratic-within-chunk cost, so total cost grows roughly linearly in T for
fixed chunk size. Decode memory stays O(1) per SSM block at any context, which is the
structural advantage the efficiency sweep is meant to show (`results/efficiency_gpu.json`).

### 29. Selective copying: why do the LTI models collapse, and how do you read the mamba3 number?

The task: train on L = 64, test on L = 128, where a few marker-value pairs must be copied past
noise. An LTI system mixes all positions with the same fixed kernel, so it cannot suppress the
noise tokens; that is the selectivity argument of the Mamba paper, and our run reproduces it:
s4d gets 0.0 sequence success and 0.064 token accuracy, the Transformer 0.0 and 0.205
(`results/selective_copy_*.json`). The selective models generalize to the doubled length:
mamba2 0.183 / 0.670, mamba 0.149 / 0.649 (sequence success / token accuracy). Mamba-3 is the
honest exception: 0.129 / 0.215. Selective copying rewards content filtering, not rotation,
and in our small educational setup the complex dynamics plus the trapezoidal rule were the
hardest to optimize (its final loss is by far the highest). We report that as measured rather
than tune it away; nothing in the Mamba-3 paper promises a win on this task at this scale.

### 30. What exactly do your tests verify, and what would a failure mean?

`tests/test_models.py`, each against a 1e-3 tolerance where numerical:

- `test_s4d_conv_equals_recurrent`: FFT-convolution form equals recurrent step form. Failure
  would mean kernel or ZOH math is inconsistent with the recurrence.
- `test_mamba2_chunked_equals_sequential`: chunked SSD equals the sequential scan for
  `chunk_size = 32` and `64`, including a padded misaligned length. Failure would mean the
  algebraic chunk unrolling is wrong.
- `test_mamba3_reduces_to_exp_euler`: with `lambda = 1` and `theta = 0`, the Mamba-3 update
  equals a hand-written Mamba-1/2 reference. Failure would mean the `beta`/`gamma` weights are
  wrong.
- `test_mamba3_alpha_is_contractive_with_rotation`: `|alpha| < 1` (stability) and nonzero
  phase (rotation present).
- `test_mamba_step_equals_forward`: Mamba-1's incremental `step()` equals the parallel
  forward. Failure would mean training and decode are different models.
- `test_classifier_shapes_and_determinism` (all five backbones): correct output shapes and
  bit-identical logits under a fixed seed.
- `test_masked_mean_ignores_padding`, `test_selective_copy_dataset_contract`,
  `test_parity_dataset_targets_are_cumulative_xor`: pooling and data-generator contracts.

### 31. What did you not implement, and what would production speed require?

Not implemented: fused selective-scan CUDA kernels and Triton SSD chunk kernels (all scans are
python or chunk-level loops), full S4 NPLR with the Woodbury-corrected kernel, the Mamba-3
parallel dual and MIMO form, a parallel associative scan for Mamba-1, hardware-aware `dt`
scaling, and incremental `step()` decode for Mamba-2/3. Production speed would require porting
the scans to fused kernels (that is what the official repositories do), accumulating state in
fp32 while running matmuls in bf16/fp16, and batching heads so the chunked form hits large
enough matmuls for tensor cores. The mathematics would not change; only the compute shape
would.

### 32. Where do your numbers come from and how do I reproduce them?

Every number in these docs is read from a committed JSON in `results/`, produced by
`python scripts/run_training.py --config configs/<task>.yaml --out results/`; each JSON stores
its data, train, and model config, environment, per-epoch history, parameter count, peak GPU
memory, and wall time, so results are auditable. Parity is 20 epochs at L = 128, selective
copying trains at L = 64 and tests at L = 128, IMDb is half-train at L = 512. At the time of
writing the IMDb runs for the mamba family are still in flight: see `results/imdb_*.json`
rather than any claim here (the completed ones are s4d 0.827 accuracy / 0.907 AUC and
transformer 0.813 / 0.894).
