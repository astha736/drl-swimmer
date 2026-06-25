#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Install script for the main project and FARMS submodules.
#
# Editable installs (-e) are used so local code changes are immediately visible
# without reinstalling the package.
# -----------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP=("$PYTHON_BIN" -m pip)

log_step() {
    echo
    echo "================================================================="
    echo "$1"
    echo "================================================================="
}

run() {
    echo "+ $*"
    "$@"
}

install_requirements() {
    local package_name="$1"
    local requirements_file="$2"

    if [[ -f "$requirements_file" ]]; then
        log_step "Installing Python requirements for: $package_name"
        echo "Requirements file: $requirements_file"
        run "${PIP[@]}" install -r "$requirements_file" --no-cache-dir
    else
        echo "Skipping requirements for $package_name: no file found at $requirements_file"
    fi
}

install_editable() {
    local package_name="$1"
    local package_path="$2"

    log_step "Installing editable package: $package_name"
    echo "Package path: $package_path"
    run "${PIP[@]}" install -e "$package_path" --no-cache-dir --no-build-isolation --config-settings editable_mode=compat
}

# -----------------------------------------------------------------------------
# 1. Install top-level build/runtime requirements
# -----------------------------------------------------------------------------
log_step "Installing build-time requirement: wheel"
echo "Reason: some packages fail to build cleanly in a fresh virtual environment without wheel."
run "${PIP[@]}" install setuptools wheel --no-cache-dir


install_requirements "main project" "$PROJECT_ROOT/requirements.txt"

# -----------------------------------------------------------------------------
# 2. Install FARMS submodules
# -----------------------------------------------------------------------------
log_step "Get FARMS submodules"
FARMS_ROOT="$PROJECT_ROOT/farms"
run git submodule update --init --recursive
export PYTHONPATH="$FARMS_ROOT/farms_core:$FARMS_ROOT/farms_mujoco:${PYTHONPATH:-}"
echo "PYTHONPATH=$PYTHONPATH"

install_requirements "farms_core" "$FARMS_ROOT/farms_core/requirements.txt"
install_editable "farms_core" "$FARMS_ROOT/farms_core"

install_requirements "farms_mujoco" "$FARMS_ROOT/farms_mujoco/requirements.txt"
install_editable "farms_mujoco" "$FARMS_ROOT/farms_mujoco"

install_requirements "farms_sim" "$FARMS_ROOT/farms_sim/requirements.txt"
install_editable "farms_sim" "$FARMS_ROOT/farms_sim"

install_requirements "farms_amphibious" "$FARMS_ROOT/farms_amphibious/requirements.txt"
install_editable "farms_amphibious" "$FARMS_ROOT/farms_amphibious"

# -----------------------------------------------------------------------------
# 3. Install the main repository
# -----------------------------------------------------------------------------

install_editable "drl-swimmer (this project)" "$PROJECT_ROOT"

log_step "Installation complete"
