# The SDPA backend is pinned per variant and detected from the profiler trace

Transformer variants name their attention backend (flash, efficient, cudnn, math, or the naive
reference), and every SDPA call runs inside sdpa_kernel([that backend]), so a backend that cannot
handle the inputs raises instead of silently falling back. The backend that actually ran is read
from the aten op names in a profiler trace of one forward pass and stored with the result.

PyTorch's default choice depends on dtype, GPU architecture and version: on sm89 it changes
between 2.14 and 2.15 (2.15 prefers cuDNN), and in float32 FlashAttention is unavailable. An
unpinned baseline therefore measures whatever the dispatcher picked. The private
torch._fused_sdp_choice reports the dispatcher's intent, not what ran; the profiler records the
kernel family that executed.

Consequences: an attention backend is part of a variant's identity, so "the Transformer" in a
table always names its backend and precision. The naive implementation is kept only as the
correctness reference and is labelled as a reference wherever it appears.
