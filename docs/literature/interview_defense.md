# Interview Defense: Copying, State Tracking, and Task Formulation

Answers are tied to the experiments in this repository (results/train/selective_copy/,
results/train/exact_copy/, results/train/exact_copy_varlen/, results/train/parity/; means
over three run seeds, five for parity, with parameter-matched models) and to the papers in
REFERENCES.md.

**Why does one benchmark favor Mamba and another favor Transformers?**
Because "copying" is not one task. Selective copying asks the model to keep k marker->value
pairs (constant information) and filter a long noise stream: the skill is selective
forgetting, which input-dependent gates give Mamba, and LTI models are bad at it (our S4D:
28.4% token accuracy in distribution, where every selective model and both Transformers
score 98.3 to 100%). Exact copying asks the model to reproduce every token: the information
grows linearly with length, there is no noise to reject, and the binding constraint becomes
state capacity. Attention re-reads the prompt, so its memory grows with context; a fixed
state cannot. Measured: trained on string lengths 1 to 32, the Transformer with RoPE copies
98.6% of the letters at length 32, against 61.4% for Mamba-3, the best SSM, and 29.0 to
35.6% for Mamba-1/2.

**What exactly is copying?**
Reproducing a substring of the input at specified output positions. The scientifically
relevant split: (a) how much information must be retained (constant k values vs the whole
L-token string), and (b) whether the model must identify what matters (marker detection) or
retain everything.

**Why does fixed-size state matter?**
An SSM's state is a fixed-size tensor, say D x N numbers. Information theory caps what can
pass through it: to reproduce L tokens from an alphabet V you need roughly L log2 V bits of
state. Once L exceeds what the state plus training can encode, exact copying must fail.
Attention is not capped: the KV cache grows with context, so re-reading is always
available. Our inference benchmark shows the other side of that trade: after a 32K prompt
the Transformer's KV cache holds 64 MiB in bf16, while the SSM states hold 0.14 to 0.39 MiB.

**Why can a Transformer retrieve arbitrary positions?**
Attention computes content-based addressing: the query at the answer position compares
against keys at every prompt position and pulls the matching values. Nothing is compressed
away, so any token remains exactly retrievable at cost O(T) per query (the quadratic
training bill is the price).

**What does a selective SSM remember?**
Whatever its gates decide to keep. delta_t controls how fast each state slot forgets; B_t
injects the current token. On selective copying it learns "hold markers, ignore noise". On
exact copying there is no ignore option, so slots are overloaded and older tokens decay.

**Why can state size become a bottleneck, and do your runs show it?**
State size N and model width D fix the state dimension, which caps what can be copied in
principle. At a single training length (32) our runs do not show the cap: the answer is a
fixed delay, and the LTI S4D and Mamba-3 copy perfectly. Over variable lengths they show a
gap consistent with it: every SSM falls behind the Transformers at length 32. Mamba-1 and
Mamba-2 also learn slowly under the shared six-epoch budget (their validation loss is still
falling at the end), so part of their gap is optimization, not capacity.

**What is a fair model comparison?**
Match the things that could otherwise explain the gap: parameter count (ours: every SSM
within 10% of the Transformer per task), training data, steps, schedule, evaluation protocol
and attention kernel; run several seeds and report the spread. Then vary one thing: the task
formulation. Also state what is not matched: our benchmark sweeps use equal width rather
than equal parameters, our Mamba has no fused kernels (wall-clock), and our Transformers use
sinusoidal or rotary positions (the paper's best extrapolating copier used ALiBi).

**What is length generalization?**
Performance on inputs longer than anything seen in training. It is a different skill from
in-distribution accuracy: it requires the solution to be expressed in a length-independent
way. None of our models has it: on exact copying every model, both Transformers included, is
at most 13.1% at lengths 64 and 128, and on parity every mean falls to 50.1 to 51.5% at
length 2048 (chance 50%). Jelassi et al. report that it is encoding-dependent for
Transformers (ALiBi extrapolates); we did not train ALiBi.

**Why can different benchmarks give opposite conclusions?**
Because each benchmark samples a different region of the skill space: memory capacity vs
selectivity vs state tracking vs noisy retrieval. A benchmark is one point in that space,
not the space. This repository measures five (selective copy, exact copy at a fixed and at
variable lengths, parity, IMDb) and the ranking changes across them.

**What did your reproduction actually prove?**
At small scale and matched parameters, with three seeds: Transformers copy variable-length
strings better than every SSM, as the paper reports; copying at a single length is a fixed
delay that even the LTI S4D solves; and no model extrapolates in length with sinusoidal or
rotary positions. It does not reproduce the paper's ALiBi result or its pretrained-model
experiments.

**If Mamba's state is the bottleneck, why did S4D copy perfectly at length 32?**
Because every training string had the same length, the answer is always the prompt delayed
by the same offset. A fixed LTI convolution implements that delay directly, with the gate
passing it through; no content selection and no length detection are needed. Trained on
lengths 1 to 32, S4D drops to 19.7% at length 32, the lowest of all models.

**Why does Mamba-3 learn parity when Mamba-1/2 do not?**
Parity needs the state to flip with each 1-bit, a rotation that real, non-negative
transitions cannot express. Mamba-3's complex transition rotates by an angle that depends on
the input through delta. Two of its five seeds learn the task (89.2 and 99.0% token
accuracy); with theta = 0 the same model never exceeds 57.2%, the level of Mamba-1/2. So the
rotation is what makes it learnable here, and whether a run finds the solution depends on
the seed.

**Rapid-fire**
- "Did Mamba fail at copying?" -> At a single length, Mamba-1/2 were still learning after
  six epochs while Mamba-3 solved it. Over variable lengths every SSM trails attention.
  Nothing extrapolates.
- "Is the paper wrong then?" -> No. The variable-length runs reproduce its ranking at small
  scale; the fixed-length runs show why the task formulation matters.
- "Which copy task is more realistic?" -> Different: selective copy is retrieval from noise;
  exact copy is duplication. LLM copying behavior (Jelassi et al. pretrained-model
  experiments) is closer to exact copying.
- "How would you fix the extrapolation?" -> Try ALiBi, as the paper does, and train on
  longer and more varied lengths.
- "What would you run next?" -> an ALiBi arm, longer training for the SSMs on exact copy,
  more seeds for Mamba-3 parity, and prefix/suffix n-gram lookup (the paper's other tasks).
