#!/bin/bash
set -e

# Install build-time requirements before FARMS. Keep wheel first because several
# packages still fail to build cleanly when wheel is missing in a fresh venv.
# Add future pre-FARMS build requirements here so the submodules see them.
pip3 install wheel --no-cache-dir
pip3 install -r requirements.txt --no-cache-dir

cd farms
cd farms_core
pip3 install -r requirements.txt --no-cache-dir
pip3 install -e . --no-cache-dir
cd ..
cd farms_mujoco
pip3 install -r requirements.txt --no-cache-dir
pip3 install -e . --no-cache-dir
cd ..
cd farms_sim
pip3 install -r requirements.txt --no-cache-dir
pip3 install -e . --no-cache-dir
cd ..
cd farms_amphibious
pip3 install -r requirements.txt --no-cache-dir
pip3 install -e . --no-cache-dir
cd ..
pip3 install -e farms_core --no-cache-dir
pip3 install -e farms_mujoco --no-cache-dir
pip3 install -e farms_sim --no-cache-dir
pip3 install -e farms_amphibious --no-cache-dir
cd ..
pip3 install -e . --no-cache-dir
