# Mamba-3's chunked form materializes per-slot segment-sum decays

The chunked Mamba-3 scan computes the intra-chunk decay exp(A_n D_ts) from a direct masked
cumsum of step sizes D_ts, materializing a (Q, Q, N) tensor per head and chunk, and bounds the
transient memory by processing chunks in groups sized from a byte budget.

The obvious faster alternative factorizes exp(A_n (cs_t - cs_s)) as exp(A_n cs_t) exp(-A_n cs_s)
and turns the intra-chunk term into a (Q, N) x (N, Q) matmul with no (Q, Q, N) tensor. Because
Re(A_n) < 0, exp(-A_n cs_s) grows with the chunk's accumulated step size and overflows float32
once |Re(A_n)| cs_s passes about 88; learned step sizes are unbounded (softplus), so no chunk size
is safe for every trained model. Step 1 already removed a difference-of-cumsums form from
Mamba-2 for losing float32 precision. Correctness is preferred over speed here because this form
is the oracle-checked path that every long-sequence Mamba-3 number goes through.

Consequences: memory per chunk grows with N, so Mamba-3's default chunk size may be smaller than
Mamba-2's. A factored form can be added later behind the same interface if a stabilized version
(per-chunk rescaling) is verified against the recurrence.
