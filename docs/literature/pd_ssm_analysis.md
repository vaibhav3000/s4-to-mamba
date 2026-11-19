# PD-SSM: Analysis Note (Research Note Only)

Paper: "Structured Sparse Transition Matrices to Enable State Tracking in
SSMs" (PD-SSM; arXiv:2509.22284, NeurIPS 2025 Spotlight). Code:
https://github.com/IBM/expressive-sparse-state-space-model (license checked:
Apache-2.0 at time of inspection).

## The idea

Mamba-style SSMs use diagonal (or scalar-times-identity) transition matrices
A_t: cheap, parallelizable, but provably weak at state tracking, because
diagonal real transitions only scale/decay coordinates and cannot implement
the rotations or permutations that finite-state automata need (the same
expressivity gap our parity experiment is consistent with: Mamba-1/2 stay near chance
while Mamba-3's input-dependent complex transitions reach 98.3% token accuracy; one seed,
pre-fix code, and no ablation that isolates the rotation). PD-SSM replaces the fully
diagonal transition with a **structured sparse** one:

    h_t = A(u_t) h_{t-1} + B(u_t) x_t,   A(u_t) = P(u_t) + D(u_t)

- **P(u_t)**: a sparse, input-dependent permutation-like component (a
  permutation swaps state coordinates, which is what tracking an FSA's state
  transitions requires),
- **D(u_t)**: a diagonal (decay) component, keeping Mamba-style gating and
  stable forgetting,
- plus a BIBO-stability-style constraint discussion so the added expressivity
  does not reintroduce instability.

Claimed result: FSA/state-tracking expressivity approaching unstructured
matrices while retaining structured-SSM parallel scan cost (the permutation
component is also a structured matrix, so the associative scan machinery
survives).

## Why it is directly relevant to our flagship

Our repository's capability axis is exactly the one PD-SSM targets:

- Mamba-1/2 (real diagonal): near chance on parity (measured 0.559/0.563 token accuracy),
- Mamba-3 (complex diagonal, input-dependent step): 0.983 token accuracy and 0.42
  sequence success, which we attribute to rotation although no theta = 0 ablation was run,
- S4D (complex LTI): chance (rotation not input-dependent),
- PD-SSM would add a third mechanism: input-dependent *permutation*.

The scientific question for a follow-up experiment is well-posed for our
harness: "can structured sparse transitions match complex transitions on
parity, and do they additionally help on tasks rotations do not cover (e.g.
permutation-heavy state tracking)?" Our parity, selective-copy and exact-copy
tasks plus the existing baseline grid would need only a new backbone file.

## Why we did not implement it now

- The scan with input-dependent permutation matrices is a genuinely new
  algorithmic component: our educational chunked SSD does not cover it, and a
  naive sequential implementation would be too slow to train the comparison
  grid on this machine within the shared budget used by the other models.
- A rushed implementation would fail the repository's testing standard (we
  would want an independent naive reference test, as tests/test_references.py has
  for S4D, Mamba-2 and Mamba-3, before benchmarking it alongside the others).
- The marginal interview value over the existing Mamba-3 complex-transition
  story is additive, not transformative: both address the same expressivity
  gap via different transition structures.

## Verdict

Documented as the SECOND extension candidate. If implemented later: minimal
sequential PD-SSM block, correctness test against a hand-computed permutation
+ decay recurrence, parity + selective-copy comparison against the existing
grid, same config/seed protocol.
