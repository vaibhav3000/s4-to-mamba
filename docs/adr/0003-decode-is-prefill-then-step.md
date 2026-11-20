# Generation is measured as a parallel prefill followed by O(1) or KV-cache steps

Every backbone exposes prefill (the parallel form over the prompt, returning the decode state)
and step (one token). Time to first token is measured through prefill, and decode cost through
step, for every architecture. This is how deployed SSMs and Transformers generate, and it is the
only setup in which "constant-time decode" and "a KV cache that grows with context" are measured
on equal terms.

The alternatives were to keep the Step 1 measurement (SSMs stepped through the whole prompt one
token at a time, the Transformer re-running its full prefix for every token) or to step the
prompt for every model. The first compared an O(1) recurrence with an O(T^2) recompute nobody
deploys; the second makes time to first token a Python-loop artifact.

Consequences: every SSM needs a prefill that returns its final state from the parallel form,
tested against step-by-step decoding in float64. The efficiency sweep's decode phase uses the same
interface and can be turned off, so the two benchmarks never disagree about what decode means.
