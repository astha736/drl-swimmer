# DRL for Swimmer Robot - AgnathaX

This repository contains the reinforcement-learning and simulation pipeline used to study feedback policies for an undulatory AgnathaX swimmer. It combines FARMS/MuJoCo simulation, a CPG-based limbless robot controller, and Stable-Baselines training for policies that modulate feedback terms such as stretch coupling and drive.

The public demo path is designed to show that the code can:

- initialize the AgnathaX swimmer in a water arena,
- run a short CPG/feedback architecture test,
- run a small PPO training job,
- save models and metrics for later inspection.

## Quick Start

Create an environment and install dependencies:

```bash
python3 -m venv env_drl_swimmer
source env_drl_swimmer/bin/activate
pip install -r requirements.txt
bash setup.sh
```

Run a headless smoke test:

```bash
python scripts/smoke_test.py
```

Run the short architecture test:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_arch_test -d demo -s 999
```

Run a tiny PPO training demo:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_short_train -d demo -s 999
```

The demo training is intentionally small and is meant to verify the pipeline, not reproduce final paper performance.

## Repository Layout

- `main.py`: experiment entry point.
- `conf.py`: global experiment configuration and logging setup.
- `rlgym/`: Gym wrapper around FARMS/MuJoCo.
- `utils/`: training, simulation setup, robot initialization, metrics, and plotting helpers.
- `agnathax_control/`: AgnathaX network/control code.
- `config/_EXPERIMENT/demo.yaml`: curated public demo experiments.
- `config/_ANIMAT/`: robot/animat configs.
- `config/_ARENA/`: arena and water configs.
- `config/_SIMULATION/`: MuJoCo/FARMS runtime configs.
- `models/`: SDF/MuJoCo model assets.
- `docs/`: pipeline and configuration notes.

## Documentation

- [Quickstart](QUICKSTART.md)
- [Architecture](docs/architecture.md)
- [Experiment configs](docs/experiment_configs.md)
- [Troubleshooting](docs/troubleshooting.md)

## Notes

Use `headless: true` for server or CI runs. Viewer mode requires a working OpenGL/display setup and may fail with GLFW window errors on headless machines.

Old paper-scale experiments and checkpoints should be archived outside the public repository. Keep only curated examples and small reproducible demo outputs in git.
