# Variants are parameter-matched by width, anchored on the Transformer

For each task, the Transformer keeps its original width and depth, and every SSM variant gets the
d_model on its constraint grid whose total parameter count is closest to the Transformer's.
Depth, state size, expansion and every training setting stay fixed. scripts/match_params.py
computes the widths (and with --check verifies the committed ones); the configs hold them as
literal numbers, and a test asserts each variant is within 10% of the anchor.

Matching by width keeps the comparison about the mixer at a fixed depth. Matching by depth would
change the number of nonlinear stages, and matching by state size would change the very capacity
the copy tasks probe. A 10% tolerance is needed because Mamba-2 and Mamba-3 widths move in steps
of headdim / expand, which at these sizes changes the count by roughly 15 to 30% per step; the
exact counts are reported next to every result.

Consequence: the SSMs are wider than in the Step 1 configs, so their results are not comparable
to the pre-fix ones even apart from the bug fixes.
