# Interview Defense: Copying, State Tracking, and Task Formulation

Answers are tied to the actual experiments in this repository
(results/selective_copy_*.json, results/exact_copy_*.json, and the papers in
REFERENCES.md).

**Why does one benchmark favor Mamba and another favor Transformers?**
Because "copying" is not one task. Selective copying asks the model to keep k
marker->value pairs (constant information, ~5 values) and filter a long noise
stream: the skill is selective forgetting, which is exactly what input-dependent
gates give Mamba, and LTI models are provably bad at it (our s4d: 6% token
accuracy). Exact copying asks the model to reproduce every token: the
information grows linearly with length, there is no noise to reject, and the
binding constraint becomes state capacity. Attention re-reads the prompt, so
its memory grows with context; a fixed state cannot. Same word, different
bottleneck, opposite ranking. Our measured flip: on selective copy mamba2
0.183 vs transformer 0.0 sequence success; on exact copy transformer 1.0
(in-dist) vs mamba2 0.0.

**What exactly is copying?**
Reproducing a substring of the input at specified output positions. The
scientifically relevant split: (a) how much information must be retained
(constant k values vs the whole L-token string), and (b) whether the model
must identify what matters (marker detection) or retain everything.

**Why does fixed-size state matter?**
An SSM's state is a fixed-size tensor, say D x N numbers. Information theory
caps what can pass through it: to reproduce L tokens from an alphabet V you
need roughly L log2 V bits of state. Once L exceeds what the state plus
training can encode, exact copying must fail. Attention is not capped: the KV
cache grows with context, so re-reading is always available.

**Why can a Transformer retrieve arbitrary positions?**
Attention computes content-based addressing: the query at the answer position
compares against keys at every prompt position and pulls the matching values.
Nothing is compressed away, so any token remains exactly retrievable at cost
O(T) per query (the quadratic training bill is the price).

**What does a selective SSM remember?**
Whatever its gates decide to keep. delta_t controls how fast each state slot
forgets; B_t injects the current token. On selective copying it learns "hold
markers, ignore noise". On exact copying there is no ignore option, so slots
are overloaded and older tokens decay.

**Why can state size become a bottleneck?**
State size N and model width D fix the state dimension. Our exact-copy result
makes the bottleneck visible: in-distribution the Mambas plateau around 0.11-0.15
token accuracy at the same budget where the Transformer is perfect, and past
the training range every SSM decays toward chance as length grows.

**What is a fair model comparison?**
Match the things that could otherwise explain the gap: parameter scale,
training data, steps, schedule, evaluation protocol (ours: identical configs,
one shared budget, seeds fixed). Then vary exactly one thing: the task
formulation. Also state what is NOT matched: our educational Mamba lacks fused
kernels (wall-clock), and our Transformer uses sinusoidal positions (paper's
best copier used ALiBi).

**What is length generalization?**
Performance on inputs longer than anything seen in training. It is a different
skill from in-distribution accuracy: it requires the solution to be expressed
in a length-independent way. Our exact-copy extrapolation shows every model
collapsing at our scale, and the paper shows it is PE-dependent for
Transformers (ALiBi extrapolates, NoPE does not) and capacity-bound for SSMs.

**Why can different benchmarks give opposite conclusions?**
Because each benchmark samples a different region of the skill space: memory
capacity vs selectivity vs state tracking vs noisy retrieval. A benchmark is
one point in that space, not the space. Our repo now measures four points
(selective copy, exact copy, parity, IMDb) and the ranking changes across
them.

**What did your reproduction actually prove?**
A qualitative reproduction at ~1/8 scale: the paper's in-distribution ranking
(transformers >> selective SSMs on exact copying) reproduces at matched
budget, its PE-dependence finding reproduces as a failure mode (our
sinusoidal-PE transformer collapses exactly where they say absolute PEs do),
and combined with our selective-copy results it establishes the
task-dependence claim. It does not reproduce their ALiBi extrapolation
success, and it is single-seed.

**If Mamba's state is the bottleneck, why did s4d copy perfectly in-distribution?**
Because its block is gated (input-dependent SiLU gate) even though the
convolution kernel is LTI. Within trained lengths the kernel acts as a bank of
learned echo responses and the gate selects the right one. Past the trained
range it collapses to chance: the echo bank has no response for unseen delays.
In-distribution copying does not require selectivity over content; it requires
a delay line plus length detection, which an LTI + gate system can implement.

**Rapid-fire**
- "Did Mamba fail at copying?" -> In-dist it lagged badly at our budget; the
  capacity failure at extrapolation is the structural result.
- "Is the paper wrong then?" -> No; both rankings are real; the tasks differ.
- "Which copy task is more realistic?" -> Different: selective copy is
  retrieval-from-noise; exact copy is duplication. LLM copying behavior
  (Jelassi et al. pretrained-model experiments) is closer to exact copying.
- "How would you fix the Transformer extrapolation?" -> ALiBi or RoPE, per the
  paper; documented future work.
- "What would you run next?" -> prefix/suffix n-gram lookup (their other
  tasks), multiple seeds, ALiBi arm.
