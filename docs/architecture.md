# Architecture

The pipeline is organized around named experiment configurations.

1. `main.py` reads a named experiment from a YAML file.
2. `conf.py` expands defaults, loads referenced PPO/reward YAML files, and creates log paths.
3. FARMS loads animat, arena, and simulation options from `config/_ANIMAT`, `config/_ARENA`, and `config/_SIMULATION`.
4. `utils/limbless_experiment_options.py` applies the robot architecture: CPG coupling, stretch feedback, initial pose, and oscillator initial state.
5. `utils/simulation.py` creates the FARMS/MuJoCo simulation and AgnathaX controller.
6. `rlgym/rl_gym.py` wraps the simulation as a Gym environment for Stable-Baselines.
7. `utils/train_test.py` trains, evaluates, and saves metrics/models.

The public demo configs live in `config/_EXPERIMENT/demo.yaml`.
