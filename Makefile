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

probe: ## Report what the installed Isaac Sim actually does
	uv run graft sim probe --config $(CONFIG)

capture: ## Render clips in Isaac Sim (headless)
	uv run graft capture --config $(CONFIG)

qa: ## Check rendered frames and record quarantines
	uv run graft qa --config $(CONFIG)

assemble: ## Build the image dataset from captured clips
	uv run graft assemble --config $(CONFIG)

train: ## Train the detector
	uv run graft train --config $(CONFIG)

eval: ## Score on sim-val and real photos
	uv run graft eval --config $(CONFIG)

pipeline-sim: init capture qa assemble train eval ## USD to trained weights, one command

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
