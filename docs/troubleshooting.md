# Troubleshooting

## Missing FARMS Packages

`setup.sh` expects local FARMS packages under `farms/`:

```text
farms/farms_core
farms/farms_mujoco
farms/farms_sim
farms/farms_amphibious
```

They are configured as private submodules in `.gitmodules`. Populate them before
installing:

```bash
git submodule update --init --recursive
bash setup.sh
```

If you do not have access to `git@ponyo.epfl.ch`, you need an accessible copy of
those FARMS repositories or preinstalled compatible packages.

## No Top-Level `requirements.txt`

This checkout does not contain a root `requirements.txt`. Use:

```bash
bash setup.sh
```

`setup.py` lists the core package dependencies, while `setup.sh` installs FARMS
subpackages plus extra tooling used by experiments.

## GLFW Or Display Errors

If you see `RuntimeError: Failed to create window`, the simulation tried to open
a viewer without a working display. Use a simulation config with:

```yaml
headless: true
```

The headless demo config is:

```text
config/_SIMULATION/simulation_headless_demo.yaml
```

Viewer demos such as `demo_caudal_feedback_viewer` and `demo_cpg_swim_viewer`
need a working display/OpenGL setup.

## Log Directory Already Exists

Runs write to:

```text
experiments/<experiment_id>/logs/<date>/<seed>
```

Existing non-debug log directories abort to avoid overwriting results. For quick
local demos, use seed `999` or choose a fresh `-d` value.

## Short Training Does Not Create `best_model.zip`

Very short training runs may finish before `EvalCallback` records a new best
model. `TrainTestClass.exp_training()` saves `last_model_trained.zip` and falls
back to saving the final model as `best_model.zip` when needed.

## Missing `utils.utils`

`rlgym/rl_gym.py` and `utils/train_test.py` import metric helpers through
`from utils import utils`, but this checkout does not include `utils/utils.py`.

Likely symptoms:

```text
ImportError: cannot import name 'utils' from 'utils'
ModuleNotFoundError: No module named 'utils.utils'
```

The smoke test may still step the simulator because it does not call the metrics
helpers. Architecture testing and trained-policy evaluation can fail when they
try to call `get_performance_metrics` or `save_performance_metrics`.

## `STRETCH_BIAS` Or Historical Reward Terms Fail

Some declared features are stubs. These raise `NotImplementedError` if enabled:

- `STRETCH_BIAS` actions
- `active_torques`
- `active_torque_diff`
- `straightness_2`
- curriculum level `1`
- PyBullet simulation

Use the demo configs as known-safe starting points.

## Stable-Baselines3 Patch Expectations

Only selected patched Stable-Baselines3 files are kept under
`stable-baselines3/`. The environment still needs a compatible Stable-Baselines3
installation. If policy loading, custom metrics, or patched PPO behavior fails,
check whether your installed package matches the patched files expected by this
repository.
