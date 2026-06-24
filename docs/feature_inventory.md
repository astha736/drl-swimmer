# Feature Inventory

This inventory reflects the repository as read from the source tree. It is meant
to help future documentation, cleanup, and experiment planning.

## Simulation And Models

- FARMS/MuJoCo simulation setup through `farms_sim` and `farms_mujoco`.
- Optional PyBullet branch is detected in `utils/simulation.py`, but PyBullet
  simulation is not implemented for this pipeline.
- AgnathaX model assets under `models/agnathax_no_tail/` and `models/raw_model.sdf`.
- Salamander, scaffold, flat-arena, water-arena, and peg geometry assets under
  `models/`.
- Water and scaffold arena configs under `config/_ARENA/`.
- Headless and viewer runtime configs under `config/_SIMULATION/`.
- Optional camera callback/video recording when `sim_options.video` is enabled.

## Robot Controller

- Cython-backed network ODE in `agnathax_control/ode.pyx`.
- FARMS network wrapper in `agnathax_control/network.py`.
- 10 body-joint limbless swimmer abstraction.
- Left/right oscillators per body segment.
- Intra-segmental anti-phase coupling.
- Inter-segmental nearest-neighbor phase-biased coupling.
- Stretch feedback through local, rostral, or caudal joint-to-oscillator maps.
- Reaction-force feedback map support.
- Sensitivity modes mapped from `COS`, `SIN`, `NONPERIOD`, and `SIGNONE`.

## Experiment Initialization

- Initial body shape and pose helpers in `utils/limbless_spawn.py`.
- Initial oscillator condition helpers in `utils/limbless_oscillator.py`.
- Domain/friction helper in `utils/limbless_domain.py`.
- Composite experiment condition objects in
  `utils/limbless_experiment_options.py`.
- Historical design-of-experiment helpers for friction, spawn conditions,
  sensitivity sweeps, open/closed CPG variants, and RL-oriented feedback setups.

## Reinforcement Learning

- Gym environment wrapper in `rlgym/rl_gym.py`.
- PPO training through Stable-Baselines3 and `CustomActorCriticPolicy`.
- SAC placeholder/config path; `SACparams` currently constructs a default
  `"MlpPolicy"` SAC model.
- Optional `VecNormalize` observation/reward normalization.
- Optional state-history observations.
- Optional observation sub-selection for compatible observation/action choices.
- Optional frame skipping through `frames_per_action`.
- Optional early termination based on oscillator phase spread.
- Reset randomization for body shape, pose, joint/link velocity, COM velocity,
  target velocity, and initial drive depending on config fields.
- Curriculum stages 2-7 that swap reward terms and reset randomization settings.

## Actions

- `DRIVE`: two drive outputs, rescaled to `[1.5, 3.0]`.
- `STRETCH`: anti-symmetric stretch feedback output.
- `STRETCH_2`: symmetric stretch feedback output.
- `CONTACT`: action handler exists for contact feedback weights, but YAML parsing
  in `main.py` does not currently expose it.
- `STRETCH_BIAS`: action-space sizing exists, but applying the action raises
  `NotImplementedError`.

## Observations

- Joint positions.
- Joint velocities.
- Right-side oscillator phases, with `sin`, `cos`, or modulo preprocessing.
- Right-side oscillator amplitudes.
- Current and target COM velocity or speed.
- Contact reaction force components and norms.
- Relative phase differences between neighboring right-side oscillators.
- Absolute phase differences relative to the first right-side oscillator.

`main.py` exposes the most-used observation groups from YAML. Some enum members
defined in `ObservationType` are implemented in `ObservationChoice` but not
currently added by `main.py`.

## Rewards And Metrics

- Forward COM velocity/speed reward.
- Target speed and target velocity penalties.
- Joint power and command torque penalties.
- Constant health bonus.
- Forward displacement rewards.
- Phase-lock and frequency-lock penalties.
- Phase-spread penalty.
- Straightness reward at episode end.
- Target tracking error logging in evaluation/test environments.
- Cross-seed aggregation script in `utils/cross_seed_evalution.py`.

The metric-save path references `utils.get_performance_metrics` and
`utils.save_performance_metrics`, but the expected `utils/utils.py` module is
missing from this checkout.

## Outputs

- Model checkpoints: `best_model.zip`, `last_model_trained.zip`.
- Normalization snapshots: `best_model_normalize.pkl`,
  `last_model_trained_normalize.pkl`.
- TensorBoard logs.
- YAML evaluation metrics.
- Single-test text metrics.
- Feedback-weight PDFs.
- Gradient PDFs/GIFs when `--log_level max` is used.
- Optional video files.

## Known Issues And Cleanup Opportunities

- `utils/utils.py` is missing even though training/evaluation imports it.
- README/quickstart previously referenced a nonexistent top-level
  `requirements.txt`; use `setup.sh` and `setup.py` instead.
- `setup.sh` assumes FARMS submodules are populated and installs additional
  dependencies with unpinned versions.
- `main.py` uses a broad `except` around config loading, so genuine YAML/key
  errors can be mistaken for the legacy config path.
- Several imports are unused, for example `from re import X`,
  `from numpy import require`, and `from torch import Value`.
- `RobotFeedbackSenstivity` and related docs/code consistently misspell
  "Sensitivity"; renaming would be breaking unless aliases are kept.
- `STRETCH_BIAS`, `active_torques`, `active_torque_diff`, `straightness_2`,
  curriculum level 1, and PyBullet support are declared but not implemented.
- `CONTACT` actions and several reaction observation variants are implemented
  below the enum/helper level but are not wired through YAML parsing in
  `main.py`.
- `utils/cross_seed_evalution.py` has a typo in the filename and uses broad
  exception handling.
- The repository ignores most of `farms/` and `stable-baselines3/`, so a new
  checkout may look incomplete unless submodules and patched files are restored.
