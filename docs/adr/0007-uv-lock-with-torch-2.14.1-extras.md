# Environments come from uv.lock, with torch pinned to 2.14.1 through cpu and cu130 extras

The project is locked with uv. torch==2.14.1 comes from the PyTorch CPU index through the cpu
extra or from the cu130 index through the cu130 extra; the two extras are declared as conflicting
so one lock holds both. Dev tools live in a dependency group. The base dependency stays
torch>=2.14,<2.15 so a plain pip install still works.

Torch 2.15 was available but is not used: it changes the default SDPA backend on this GPU
architecture (sm89) to cuDNN and stops counting cuBLAS workspaces in allocated memory, so latency
and memory numbers would not be comparable across the upgrade. Every Step 2 number is produced
with the locked 2.14.1+cu130 build.

Consequence: moving to 2.15 or later means re-running every benchmark, and is its own change.
