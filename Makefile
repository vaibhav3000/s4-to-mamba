# Developer tasks for s4-to-mamba. Works with GNU Make on Linux and from Git Bash on
# Windows; every Python tool runs through uv in the project environment.
#
#   make setup TORCH=cu130   install the locked environment (TORCH=cpu by default)
#   make check               the full CPU gate: lint, format, types, tests, tables, lock
#   make report              rebuild the LaTeX report PDF
#   make gpu-plan            run every GPU job (needs a CUDA machine and the cu130 extra)

TORCH ?= cpu
SEEDS ?= 0 1 2
RUN := uv run --no-sync
REPORT := Efficient_Sequence_Modeling_S4_to_Mamba3
# The CPU gate hides the GPU so it runs the same everywhere.
CPU := CUDA_VISIBLE_DEVICES=-1

.PHONY: help setup lint format typecheck test tables tables-check lock-check check figures report demo gpu-plan docker

help:
	@echo "targets: setup lint format typecheck test tables tables-check lock-check check figures report demo gpu-plan docker"
	@echo "variables: TORCH=$(TORCH) (cpu or cu130), SEEDS=$(SEEDS)"

setup:
	uv sync --locked --extra $(TORCH)

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff format .

typecheck:
	$(RUN) pyright

test:
	$(CPU) $(RUN) pytest -q -p no:cacheprovider

tables:
	$(RUN) python scripts/make_tables.py

tables-check:
	$(RUN) python scripts/make_tables.py --check

lock-check:
	uv lock --check

check: lint typecheck test tables-check lock-check

figures:
	$(RUN) python scripts/make_figures.py

report:
	cd reports && latexmk -pdf -interaction=nonstopmode -halt-on-error $(REPORT).tex && latexmk -c $(REPORT).tex

demo:
	$(RUN) python demo-site/build.py

gpu-plan:
	$(RUN) python scripts/run_gpu_plan.py --seeds $(SEEDS)

docker:
	docker build -t s4-to-mamba .
	docker run --rm s4-to-mamba
