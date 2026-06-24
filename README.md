# Deep reinforcement learning for AgnathaX

This repository contains a reinforcement-learning and simulation pipeline for an
undulatory AgnathaX swimmer. It combines FARMS/MuJoCo simulation, an
oscillator-based limbless controller, YAML-defined experiment presets, and
Stable-Baselines3 training/evaluation code for policies that modulate feedback
terms such as stretch coupling and drive.

The current public/demo path can:

- initialize AgnathaX or scaffold-style model assets in flat or water arenas,
- run analytical CPG and stretch-feedback architecture tests,
- wrap FARMS/MuJoCo as a Gym environment,
- train PPO or SAC-style policies from YAML experiment definitions,
- evaluate trained policies and write metrics, model checkpoints, TensorBoard
  logs, plots, and optional videos.

## Repository Status

This is a research-code repository. The core experiment flow is present, but a
plain public checkout is not fully self-contained:

- FARMS packages are expected under `farms/` as private submodules listed in
  `.gitmodules`.
- A small patched subset of `stable-baselines3/` is kept in this tree; the rest
  of Stable-Baselines3 is expected from your Python environment.
- There is no top-level `requirements.txt`; install dependencies through the
  local FARMS packages, `setup.py`, and the extra packages listed in `setup.sh`.
- `utils/utils.py` is referenced by evaluation code but is not present in this
  checkout. Smoke tests that only step the simulator may still work, but
  architecture testing/evaluation can fail when performance metrics are saved.

## Quick Start

Clone with submodules, or initialize them after cloning:

```bash
git submodule update --init --recursive
```

Create an environment and install local packages:

```bash
python3 -m venv obstacle
source obstacle/bin/activate
bash setup.sh
```

Run a short headless simulator smoke test:

```bash
python scripts/smoke_test.py
```

Run a headless architecture test:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_arch_test -d demo -s 999
```

Run a tiny PPO training demo:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_drl_short_train -d demo -s 999
```

The demo training is intentionally small. It checks that the pipeline runs; it
is not expected to reproduce paper-quality swimming.

## Demo Experiments

Public demos live in `config/_EXPERIMENT/demo.yaml`.

- `demo_arch_test`: headless CPG/stretch-feedback architecture check.
- `demo_arch_viewer`: viewer version of the architecture check.
- `demo_caudal_feedback_viewer`: bio-inspired caudal stretch-feedback demo.
- `demo_cpg_swim_viewer`: open-loop inter-segmental CPG swimming demo.
- `demo_drl_short_train`: tiny PPO stretch-feedback training smoke demo.
- `demo_drl_target_speed_quick`: short target-speed PPO run with normalization.
- `demo_drl_target_speed_full`: longer target-speed training/evaluation preset.
- `demo_drl_fast_swim_full`: longer fast-swimming training/evaluation preset.

Viewer demos require a working OpenGL/display setup. On servers or CI, prefer
the headless configs.

## How Experiments Work

Experiments are named YAML entries. `main.py` loads one entry, `conf.py` expands
defaults and referenced reward/PPO/SAC files, FARMS loads the animat/arena/
simulation configs, and `utils/limbless_experiment_options.py` applies the
robot architecture before `utils/train_test.py` runs training, testing, or
architecture evaluation.

Supported `run_type` values:

- `arch_testing`: run a fixed CPG/feedback architecture without a learned model.
- `train`: train a Stable-Baselines model, optionally followed by evaluation.
- Testing a saved model is triggered by passing `--base_test_path` with an
  experiment id.

## RL Interface

Action choices are composed in the order defined by `main.py`:

- `DRIVE`: two oscillator drive values, rescaled to `[1.5, 3.0]`.
- `STRETCH`: anti-symmetric stretch feedback weights.
- `STRETCH_2`: symmetric stretch feedback weights.
- `STRETCH_BIAS`: declared but currently raises `NotImplementedError`.

Observation choices include:

- `VELOCITIES`: current and target COM velocity or speed.
- `JOINT_POSITION`: body joint positions.
- `PHASES`: right-side oscillator phases with `sin`, `cos`, or modulo
  preprocessing.
- `AMPLITUDES`: right-side oscillator amplitudes.
- `JOINT_VEL`: body joint velocities.
- `REACTION_XY`: contact-force norm in the swimming plane.
- `PHASE_DIFF_REL`: relative phase differences between neighboring oscillators.
- `PHASE_DIFF_ABS`: phase differences relative to the first oscillator.

Reward YAML files can combine terms for forward COM speed, target speed or
velocity tracking, joint power, health bonuses, phase locking, oscillator phase
spread, and straightness. Some historical reward keys are left as stubs and
raise `NotImplementedError` if enabled.

## Outputs

Runs write under:

```text
experiments/<experiment_id>/logs/<date>/<seed>/
```

Typical outputs include `best_model.zip`, `last_model_trained.zip`,
VecNormalize `.pkl` files when normalization is enabled, TensorBoard events,
`eval_metrics.yaml`, `single_test_env_metrics.txt`, PDF plots, and optional
recorded videos. Seed `999` is treated as a debug/demo seed and may reuse an
existing log directory.

## Repository Layout

- `main.py`: command-line experiment entry point.
- `conf.py`: global experiment configuration, defaults, and log paths.
- `rlgym/`: Gym wrapper around FARMS/MuJoCo plus actions, observations, rewards,
  callbacks, reset randomization, and evaluation hooks.
- `utils/train_test.py`: Stable-Baselines3 training, testing, model saving, and
  cross-seed aggregation.
- `utils/simulation.py`: FARMS/MuJoCo simulation assembly with the AgnathaX ODE
  controller.
- `utils/limbless_*.py`: robot architecture, oscillator, spawn, network, and
  domain initialization helpers.
- `agnathax_control/`: Cython-backed AgnathaX controller/network code.
- `config/_EXPERIMENT/`: named experiment presets.
- `config/_ANIMAT/`, `config/_ARENA/`, `config/_SIMULATION/`: FARMS model,
  arena, and runtime configs.
- `config/PPO_params/`, `config/SAC_params/`, `config/RewardFnc_params/`: RL
  hyperparameter and reward-weight files.
- `models/`: SDF/MuJoCo model assets and meshes.
- `docs/`: architecture, configuration, feature, and troubleshooting notes.

## Documentation

- [Quickstart](QUICKSTART.md)
- [Architecture](docs/architecture.md)
- [Experiment configs](docs/experiment_configs.md)
- [Feature inventory](docs/feature_inventory.md)
- [Troubleshooting](docs/troubleshooting.md)
