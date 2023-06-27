#!/bin/bash
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
pip install tensorboard black ffmpeg gym onnx graphviz torchview --no-cache-dir
# pip install sb3-contrib
# pip install stable-baselines3 # install in editable mode manually