# Training runs use deterministic algorithms

Training sets torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG=:4096:8 before
CUDA initializes, so a run seed reproduces its result exactly on the same machine and an
interrupted run resumes to bitwise the same weights.

Without it, the memory-efficient SDPA backward and some reductions are nondeterministic, and the
seed spread would mix optimization variance with kernel nondeterminism. The cost is speed: cuDNN
attention is disabled and some kernels take slower deterministic paths. Benchmarks do not enable
it, because they measure the kernels a deployment would use.

Consequence: an op without a deterministic CUDA implementation fails loudly; the smoke job runs
every variant for two steps on the GPU before any long job to find such ops early.
