# Legacy code from the original SSM project

These are the two scripts committed to the original repository
(github.com/vaibhav3000/projects/tree/main/SSM, `Code/` folder), preserved
verbatim for provenance. They are kept for the record, not for use.

Why they were replaced:

- Both scripts reference a `Mamba` / `S4Block` class that is never imported or
  defined, so neither runs as committed.
- There is no Transformer or S4 implementation anywhere in the original repo,
  although the README and CV claimed both were "implemented and benchmarked".
- The reported numbers (IMDb 0.96 vs 0.94 AUC, 50% training-time reduction)
  have no result artifacts behind them in the repository.
- The README also claimed GBT discretization implementations (Forward Euler,
  Backward Euler, Trapezoidal) that exist only as math in the accompanying
  course report, not as code.

The replacement study (this repository) implements everything it benchmarks,
verifies parallel and recurrent forms against each other in tests, and
commits every metric as JSON under `results/`.
