# Quickstart

This project expects the FARMS repositories in `farms/`. They are configured as
private submodules in `.gitmodules`, so make sure they are available before
installing:

```bash
git submodule update --init --recursive
```

Create a virtual environment and install the local packages:

```bash
python3 -m venv obstacle
source obstacle/bin/activate
bash setup.sh
```

There is no top-level `requirements.txt` in this checkout. `setup.sh` installs
the FARMS packages, this package, and extra tooling such as TensorBoard, Gym,
ONNX, Graphviz, TorchView, and scikit-learn.

## Headless Checks

Run a short FARMS/MuJoCo stepping smoke test:

```bash
python scripts/smoke_test.py
```

Run the headless CPG/stretch-feedback architecture demo:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_arch_test -d demo -s 999
```

Run a tiny PPO training demo that learns stretch-feedback weights from an
initially open/decoupled network:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_drl_short_train -d demo -s 999
```

The training demo is deliberately tiny. It verifies the training code path and
checkpoint saving, not final swimming performance.

## Viewer Demos

These need a working display/OpenGL setup:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_caudal_feedback_viewer -d demo -s 999
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_cpg_swim_viewer -d demo -s 999
```

On a headless server, use `demo_arch_test` or another config that points to
`config/_SIMULATION/simulation_headless_demo.yaml`.

## Testing A Saved Run

To evaluate an existing run directory, pass both the experiment id and
`--base_test_path`:

```bash
python main.py \
  -c config/_EXPERIMENT/demo.yaml \
  -e demo_drl_target_speed_full \
  -m experiments/demo_drl_target_speed_full/logs/<date>/<seed>
```

The saved run directory should contain the trained model files expected by
`utils/train_test.py`, usually `best_model.zip` and, when `normWrapper: true`,
`best_model_normalize.pkl`.
