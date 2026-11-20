# Interview Defense: Copying, State Tracking, and Task Formulation

> **Status (November 2026).** This note was written for the earlier single-seed version
> of this repository, whose models had defects that have since been fixed. The numbers and
> result files it quotes are kept for the record (the files are in the git history before
> commit `ab29379`) and are superseded by the three-seed results in `results/SUMMARY.md`,
> the README and the report.

Answers are tied to the actual experiments in this repository
(results/selective_copy_*.json, results/exact_copy_*.json, and the papers in
REFERENCES.md). Those runs are single-seed and predate later fixes to the models (see
results/PROVENANCE.md), so the numbers are stale until re-run.

**Why does one benchmark favor Mamba and another favor Transformers?**
Because "copying" is not one task. Selective copying asks the model to keep k
marker->value pairs (constant information: 3-5 values in training, 5-10 at test) and filter a long noise
stream: the skill is selective forgetting, which is exactly what input-dependent
gates give Mamba, and LTI models are provably bad at it (our s4d: 6% token
accuracy). Exact copying asks the model to reproduce every token: the
information grows linearly with length, there is no noise to reject, and the
binding constraint becomes state capacity. Attention re-reads the prompt, so
its memory grows with context; a fixed state cannot. Same word, different
bottleneck, different ranking. Our measured difference: on selective copy mamba2
0.183 vs transformer 0.0 sequence success, measured out of distribution (L=128 and 5-10
pairs after training on L=64 and 3-5 pairs); on exact copy transformer 1.0 vs mamba2 0.0 at
the single training length 32, where the LTI s4d also scored 1.0. So the exact-copy gap in
our runs is not evidence of a state-capacity limit: Mamba-1/2 were still improving when
training stopped.

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
State size N and model width D fix the state dimension, which caps what can be copied in
principle. Our exact-copy runs do not show that cap. At L=32 the Mambas reached only
0.11-0.15 token accuracy where the Transformer was perfect, but their validation loss was
still falling every epoch when the 6-epoch schedule ended, their states hold far more
numbers than a 32-letter string needs, and the LTI s4d, with fewer parameters (95,902
against 358,174 for Mamba-1), copied perfectly. Past the training length every model, the
Transformer included, falls to or near chance, and with a single training length that says
little about capacity.

**What is a fair model comparison?**
Match the things that could otherwise explain the gap: parameter scale,
training data, steps, schedule, evaluation protocol (ours: identical data, steps, schedule
and evaluation, and matched width and depth, all with one seed, 42). Then vary exactly one
thing: the task formulation. Also state what is NOT matched or is limited: parameter counts
(on exact copy, from 95,902 for s4d to 602,782 for the Transformer), a single training
string length (32), our educational Mamba lacks fused kernels (wall-clock), and our
Transformer uses absolute sinusoidal positions (paper's best copier used ALiBi).

**What is length generalization?**
Performance on inputs longer than anything seen in training. It is a different
skill from in-distribution accuracy: it requires the solution to be expressed
in a length-independent way. Our exact-copy extrapolation shows every model
collapsing at our scale after training on a single string length (32), which gives no model
a reason to learn a length-independent solution. The paper reports that it is PE-dependent
for Transformers (ALiBi extrapolates, NoPE does not) and capacity-bound for SSMs; our runs do
not test either point directly.

**Why can different benchmarks give opposite conclusions?**
Because each benchmark samples a different region of the skill space: memory
capacity vs selectivity vs state tracking vs noisy retrieval. A benchmark is
one point in that space, not the space. Our repo now measures four points
(selective copy, exact copy, parity, IMDb) and the ranking changes across
them.

**What did your reproduction actually prove?**
Less than the original write-up claimed. At ~1/8 scale and matched width and depth (not
parameter count), the Transformer beat Mamba-1/2 on in-distribution exact copying, which is
consistent with the paper's ranking. But training used one string length, Mamba-1/2 were
still improving when training stopped, and the LTI s4d also copied perfectly, so the run does
not show the paper's state-capacity mechanism. Our sinusoidal-PE Transformer collapsed past
the training length, consistent with the paper's finding about absolute PEs. Combined with
our selective-copy results this suggests rankings depend on the task, though the two scores
compare an out-of-distribution test with an in-distribution one. It does not reproduce
their ALiBi extrapolation success, it is single-seed, and it predates later code fixes.

**If Mamba's state is the bottleneck, why did s4d copy perfectly in-distribution?**
Because every training string has the same length (32), the answer is always the prompt
delayed by the same fixed offset. A fixed LTI convolution can implement that delay
directly, with the SiLU gate passing it through; no content selection and no length detection
are needed. Past the trained length the required delay changes and s4d falls to chance.
The premise is also doubtful: these runs do not show that Mamba's state is the bottleneck
(see the state-size answer above).

**Rapid-fire**
- "Did Mamba fail at copying?" -> In-dist it lagged badly within 6 epochs and was
  still improving when training stopped. Every model, the Transformer included, failed past
  the single training length, so the extrapolation results do not isolate a capacity limit.
- "Is the paper wrong then?" -> No. Our exact-copy runs are consistent with its ranking, the
  tasks differ, and neither of our copy results is clean evidence on its own (see above).
- "Which copy task is more realistic?" -> Different: selective copy is
  retrieval-from-noise; exact copy is duplication. LLM copying behavior
  (Jelassi et al. pretrained-model experiments) is closer to exact copying.
- "How would you fix the Transformer extrapolation?" -> ALiBi or RoPE, per the
  paper; documented future work.
- "What would you run next?" -> training on variable string lengths (5 to 32
  and beyond), training the SSMs to convergence, multiple seeds, an ALiBi arm, and
  prefix/suffix n-gram lookup (their other tasks).
