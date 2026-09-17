# LongMamba: Analysis Note (No Reproduction)

Paper: "LongMamba: Enhancing Mamba's Long Context Capabilities via
Training-Free Receptive Field Enlargement" (GATECH-EIC/LongMamba;
ICLR-style work on length extrapolation for Mamba without retraining).
Repository inspected: https://github.com/GATECH-EIC/LongMamba

## Why analysis only

The official experiments target LLaMA-style Mamba/ Codestral-scale models on
long-context benchmarks (LongBench-class datasets, 4k-32k+ contexts), with
multi-GPU evaluation requirements and heavyweight dependencies. On a 6 GB
RTX 4050 the full pipeline is impractical, and the paper's contribution is a
training-free intervention on pretrained models we do not host. Reproducing
it would violate the "no fabricated reproduction" rule, so this note records
the mechanism and its connection to our benchmarks instead.

## The problem it addresses

Mamba-family models degrade when inference contexts exceed the training
distribution. LongMamba traces part of the failure to **global channels**:
some state channels aggregate long-range summary information whose effective
receptive field grows with sequence length; at unseen lengths these channels
produce out-of-distribution activations that corrupt decoding.

## The mechanism (as we understand it)

1. Identify global channels empirically at test time.
2. **Memory decay recalibration**: rescale the decay/forgetting behavior of
   those channels so their effective receptive field matches the new length.
3. **Token filtering**: drop low-importance tokens' contributions to the
   global channels (reducing their receptive field), bounding their
   distribution shift.

No retraining: it is an inference-time correction, similar in spirit to
context-extension tricks for attention models (position interpolation, YaRN)
but applied to the SSM state dynamics instead of positional embeddings.

## Why it matters to this repository

Our exact-copy extrapolation results (results/exact_copy_*.json) show every
model collapsing outside the training length distribution, and our
efficiency suite shows what happens near hardware limits. LongMamba's thesis
is that length extrapolation failures have *mechanical* causes inside the
state dynamics (specific channels, specific decays), not just "not enough
training". That complements our Jelassi-et-al. reproduction: the paper shows
Transformers win copying via dynamic retrieval; LongMamba asks whether SSM
length failures can be patched at test time. A cheap follow-up in our style
would be measuring per-channel activation drift in our minimal Mamba as test
length grows (we can log state norms per channel); the full training-free
recalibration on pretrained Mambas stays out of scope for this 6GB machine.

## Status

Analysis note only. No reproduction attempted; nothing here should be cited
as a replication of LongMamba's results.
