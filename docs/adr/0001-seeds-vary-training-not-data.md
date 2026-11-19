# Run seeds vary training, not the datasets; validation is separate from test

A run seed sets parameter initialization, dropout and the order of training batches. The
training, validation and test examples are fixed by the data seed and are the same for every run
and every model variant, so the spread across seeds measures optimization variance on one fixed
task instance, and every variant is scored on identical test examples. Resampling the data per
seed would also capture sampling variance, but would make paired comparisons between variants
impossible and change what a table row means.

The per-epoch history is computed on a validation set drawn from the training distribution with
its own data seed (for IMDb, 2,500 training reviews held out from the training subset). Test
evaluation sets are scored once, after the last epoch. There is no early stopping or checkpoint
selection, so the test sets never influence which model is reported.

Consequence: a reported "± std" is over run seeds only. A claim that depends on the particular
test examples is not covered by it.
