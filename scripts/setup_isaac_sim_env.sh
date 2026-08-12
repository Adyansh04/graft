#!/usr/bin/env bash
# Sets up a dedicated venv for Isaac Sim / Replicator code. Separate from
# the core env (scripts/setup_core_env.sh) for DEPENDENCY isolation, not
# Python version — both are 3.12 since Isaac Sim 6.0. Specifically:
#   - usd-core (core env) ships its own compiled pxr that collides with
#     Isaac's bundled one at the Boost.Python level. Never install it here.
#   - ultralytics' torch collides with isaacsim's.
# graft itself is never installed here; sim stages run via PYTHONPATH=src.
#
# Isaac Sim's Python pin moves per release (3.10 for 4.x, 3.11 for 5.x,
# 3.12 for 6.0.1 as of 2026-08-12). Re-verify at
# https://docs.isaacsim.omniverse.nvidia.com before bumping the version.
# Needs ~25GB of disk for isaacsim + extscache.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ISAAC_SIM_PYTHON_VERSION="3.12"
ISAAC_SIM_INDEX="https://pypi.nvidia.com"

uv venv --clear .venv-isaac --python "${ISAAC_SIM_PYTHON_VERSION}"

# isaacsim's dependencies straddle pypi.nvidia.com and PyPI — some packages
# exist on NVIDIA's index but only at versions isaacsim does not want. uv's
# default is to resolve a package from the first index that has it, which
# makes those unsatisfiable, so it has to consider both.
# Isaac Sim prompts for the Omniverse EULA on first launch and fails with
# EOF when run non-interactively. Accepted by the repository owner on
# 2026-08-12; override by exporting OMNI_KIT_ACCEPT_EULA=N.
# https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# The extension-cache wheels are multi-GB and exceed uv's default 30s HTTP
# timeout on an average connection.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1800}"

# --prerelease=allow: some isaacsim dependencies are pinned to release
# candidates (tinyobjloader==2.0.0rc13).
uv pip install --python .venv-isaac \
    --extra-index-url "${ISAAC_SIM_INDEX}" \
    --index-strategy unsafe-best-match \
    --prerelease=allow \
    -r scripts/requirements-isaac.txt
echo "Isaac Sim env ready: .venv-isaac/ (Python ${ISAAC_SIM_PYTHON_VERSION})"
echo "Verify with: uv run graft doctor"
