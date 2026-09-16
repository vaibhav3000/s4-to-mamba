# References

Papers are the source of the ideas; the official repositories were consulted for
conventions only. The relationship between this repository and each source is stated per
entry, and the originality statement at the end applies to all of them.

## Papers

- **HiPPO: Recurrent Memory with Optimal Polynomial Projections.** Gu, Dao, Ermon, Rudra,
  Re (2020). [arXiv:2008.07669](https://arxiv.org/abs/2008.07669)
  The memory theory behind structured state space initialization. It motivates the
  initialization of `A` in `src/ssmbench/models/s4_minimal.py` and is the root of the whole
  line of work traced in `docs/evolution.md`.

- **Combining Recurrent, Convolutional, and Continuous-time Models with Linear State Space
  Layers (LSSL).** Gu, Johnson, Goel, Saab, Dao, Rudra, Re (2021).
  [arXiv:2110.13985](https://arxiv.org/abs/2110.13985)
  The template that unifies the recurrence, convolution, and ODE views of the same layer;
  the framing used throughout `docs/evolution.md`.

- **Efficiently Modeling Long Sequences with Structured State Spaces (S4).** Gu, Goel, Re
  (2021). [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
  The NPLR parameterization, HiPPO initialization, and the Woodbury-corrected convolution
  kernel. Referenced, not reimplemented: this repository implements the S4D diagonal
  restriction only, as `docs/evolution.md` states plainly.

- **On the Parameterization and Initialization of Diagonal State Space Models (S4D).** Gu,
  Gupta, Goel, Re (2022). [arXiv:2206.11893](https://arxiv.org/abs/2206.11893)
  The model implemented in `src/ssmbench/models/s4_minimal.py`: complex diagonal `A` with
  S4D-Lin initialization (`A_n = -1/2 + i pi n`), ZOH discretization with the exact
  `(e^z - 1)/z` factor, FFT-convolution training form, and O(1)-state recurrent form.

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Mamba-1).** Gu, Dao
  (2023). [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
  Selectivity (input-dependent `delta`, `B`, `C`), the block structure with short convolution
  and gating implemented in `src/ssmbench/models/mamba_minimal.py`, and the dt bias
  initialization convention.

- **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured
  State Space Duality (Mamba-2).** Dao, Gu (2024).
  [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
  Scalar-per-head transitions and the chunked SSD algorithm implemented (both chunked and
  sequential forms, tested equal) in `src/ssmbench/models/mamba2_minimal.py`.

- **Mamba-3: Improved Sequence Modeling using State Space Principles.** Lahoti, Li, Chen,
  Wang, Bick, Kolter, Dao, Gu (2026).
  [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
  The three changes implemented or described in `src/ssmbench/models/mamba3_minimal.py`:
  exponential-trapezoidal discretization (its Table 1 also names the exponential-Euler rule
  used retroactively for Mamba-1/2), complex-valued transitions, and the MIMO formulation
  (described only; this repository keeps SISO per head). Its expressivity discussion, drawing
  on Grazzi et al. and Merrill et al., is the background for the parity result in
  `results/parity_*.json`.

## Code

- **state-spaces/mamba** (official Mamba, Mamba-2 and Mamba-3 reference code, Apache-2.0
  license). [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
  Consulted for parameterization conventions and the dt bias initialization (log-uniform in
  `[1e-3, 1e-1]` through an inverse-softplus bias, replicated in the `_init_dt` methods here).
  The official package builds CUDA extensions and requires Linux; this repository
  reimplements the ideas educationally in pure PyTorch so that everything runs on any
  platform and can be read top to bottom.

- **state-spaces/s4** (official S4 and S4D reference code, Apache-2.0 license).
  [https://github.com/state-spaces/s4](https://github.com/state-spaces/s4)
  Consulted for S4D parameterization conventions (log-space real part of `A`, complex `B` and
  `C`, skip connection `D`) used in `src/ssmbench/models/s4_minimal.py`.

## Originality statement

No code was copied from the official repositories or from any other source. Ideas, equations,
and initialization conventions are cited to the papers above, and the official repositories
were used as reading material for conventions, not as code to transplant. Every file in
`src/ssmbench/models/` is an original educational implementation written from first
principles; correctness is established by the equivalence tests in `tests/test_models.py`
(parallel versus recurrent forms, chunked versus sequential scans, and reduction of Mamba-3
to the Mamba-1/2 rule), not by comparison against reference kernels. Benchmark numbers quoted
in the documentation come exclusively from committed JSON files in `results/`.
