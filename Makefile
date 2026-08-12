CONFIG ?= configs/default.yaml

.PHONY: help setup setup-isaac doctor test validate init status clean-pyc

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the core venv and install deps
	./scripts/setup_core_env.sh

setup-isaac: ## Create the Isaac Sim venv (needs ~25GB free)
	./scripts/setup_isaac_sim_env.sh

doctor: ## Check the environment is sane
	uv run graft doctor

test: ## Run the pure-Python test suite (no Isaac Sim needed)
	uv run pytest -q

validate: ## Validate the config without running anything
	uv run graft validate --config $(CONFIG)

init: ## Create the run directory and snapshot the config
	uv run graft run init --config $(CONFIG)

status: ## Show stage and clip progress
	uv run graft status --config $(CONFIG)

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
