#!/bin/bash
cd farms
cd farms_core
pip3 install -r requirements.txt
pip3 install .
cd ..
cd farms_mujoco
pip3 install -r requirements.txt
pip3 install .
cd ..
cd farms_sim
pip3 install -r requirements.txt
pip3 install .
cd ..
cd farms_amphibious
pip3 install -r requirements.txt
pip3 install .
cd ..
pip3 install -e farms_core
pip3 install -e farms_mujoco
pip3 install -e farms_sim
pip3 install -e farms_amphibious
cd ..
pip3 install .