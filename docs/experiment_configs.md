# Experiment Configs

Named experiments are YAML entries. Run one with:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_arch_test -d demo -s 999
```

The default config path is `config/_EXPERIMENT/conf.yaml`, which contains
curated historical/research presets. Public smoke and demo runs live in
`config/_EXPERIMENT/demo.yaml`.

## Top-Level Fields

- `run_type`: `arch_testing` for a fixed CPG/feedback simulation, or `train` for
  Stable-Baselines training.
- `robot_arch`: oscillator coupling and feedback initialization.
- `config`: FARMS animat, arena, simulation, and optional drive/timestep values.
- `RL`: action, observation, reward, policy, and reset/randomization settings.
- `frames_per_action`: number of simulator steps applied for each policy action.
- `n_iterations`: training episode length in simulator iterations.
- `n_iterations_testing`: architecture/evaluation episode length.
- `training`: optional overrides for evaluation frequency and evaluation
  episode count during training.
- `post_training.run_test`: whether to run `TrainTestClass.test()` after
  training.
- `evaluation`: evaluation count and cross-seed aggregation behavior.

## `robot_arch`

Important fields:

- `init_osci_cond`: oscillator initial condition selector. `0` is ideal, `1` is
  random, and `-1` is a fixed preset according to the code comments.
- `c_inter`: inter-segmental oscillator coupling weight. Use `0` for decoupled
  CPGs, positive values for nearest-neighbor wave coupling.
- `s_caudl_weight`: caudal stretch-feedback initialization.
- `s_local_weight`: local stretch-feedback initialization.
- `s_caudl_senstivity`: one of `COS`, `SIN`, `NONPERIOD`, `SIGNONE`.
- `s_local_senstivity`: one of `COS`, `SIN`, `NONPERIOD`, `SIGNONE`.

Exactly one of `s_caudl_weight` and `s_local_weight` must be non-null. This is
validated in `conf.py`.

## `config`

- `animat`: path to an animat/model/control YAML.
- `arena`: path to an arena/water YAML.
- `simulation`: path to a simulation runtime YAML.
- `drive`: optional initial value for both oscillator drives. Defaults to `2.5`
  in `main.py`.
- `timestep`: optional simulation timestep override.

Use `config/_SIMULATION/simulation_headless_demo.yaml` or another config with
`headless: true` on servers.

## `RL`

Core fields:

- `normWrapper`: enable `VecNormalize`.
- `RewardFnc`: path to a reward-weight YAML.
- `PPOparams` or `SACparams`: path to algorithm hyperparameters.
- `episodes_per_training`: multiplied by `n_iterations` to compute total
  training timesteps.
- `action_choice`: ordered list of action groups.
- `observation_choice`: ordered list of observation groups.
- `phase_preprocessing`: required when `PHASES` is observed; values are `sin`,
  `cos`, or `mod`.
- `policy_network.arch`: hidden-layer sizes for the custom actor network.
- `policy_network.act_fn`: PyTorch activation class name, such as `Tanh`.
- `value_network`: optional value-network config; defaults to `policy_network`.
- `target_speed` or `target_velocity`: used by velocity observations and target
  tracking rewards.
- `useEarlyTerm`: terminate non-test episodes when oscillator phase spread grows
  too large.
- `useRandStartCond`: selects body/velocity reset randomization mode.
- `useRandStartCondPhases`: consumed by oscillator reset code.
- `randomInitDrive`: optional `[min, max]` range for random initial drive.
- `sample_target_velocity_from_speed_range`: optional `[min, max]` speed range
  for sampled target velocities.
- `sample_init_velocity_from_speed_range`: optional `[min, max]` speed range for
  initial COM velocity.
- `curriculum`: optional multi-stage reward/reset curriculum.
- `state_history_length`: optional temporal observation stack.
- `selectObs`: optional index subset for each 10-element observation group.

`selectObs` cannot currently be combined with `VELOCITIES` observations or
active `DRIVE` actions; `conf.py` raises an error for that combination.

## Actions

Actions are assembled in this fixed order in `main.py`:

1. `DRIVE`
2. `STRETCH`
3. `STRETCH_BIAS`
4. `STRETCH_2`

Available action groups:

- `DRIVE`: two values, rescaled from `[-1, 1]` to `[1.5, 3.0]`.
- `STRETCH`: one value per local joint, or nine values for caudal feedback; left
  and right oscillator weights are anti-symmetric.
- `STRETCH_2`: same shape as `STRETCH`, but left and right oscillator weights
  are symmetric.
- `STRETCH_BIAS`: declared but not implemented.
- `CONTACT`: implemented in `ActionChoice`, but not currently wired from
  `main.py` config parsing.

## Observations

Observations are assembled in this fixed order in `main.py`:

1. `VELOCITIES`
2. `JOINT_POSITION`
3. `PHASES`
4. `AMPLITUDES`
5. `JOINT_VEL`
6. `REACTION_XY`
7. `PHASE_DIFF_REL`
8. `PHASE_DIFF_ABS`

`ObservationType` also defines `REACTION_X`, `REACTION_Y`, `REACTION_Z`, and
`REACTION_XYZ`, but `main.py` does not currently add them from YAML.

## Rewards

Reward YAML files are dictionaries of weighted terms. Implemented keys include:

- `vel_com`
- `joints_power`
- `forward_x`
- `cmd_torques`
- `healthy`
- `speed_error`
- `forward_com`
- `velocity_error`
- `x_com_vel`
- `x_vel_target`
- `phase_lock`
- `frequency_lock`
- `straightness`
- `y_pos_penalty`
- `phase_spread`

Historical curriculum reward files also use aliases such as `joints_power_1`,
`joints_power_2`, `healthy_1`, and `healthy_2`; curriculum reset code maps these
back to active reward keys for each stage.

Known stubs:

- `active_torques`
- `active_torque_diff`
- `straightness_2`

These raise `NotImplementedError` if enabled.

## Demo Presets

- `demo_arch_test`: headless architecture check, two evaluation episodes.
- `demo_arch_viewer`: viewer architecture check.
- `demo_caudal_feedback_viewer`: caudal stretch-feedback viewer demo.
- `demo_cpg_swim_viewer`: open-loop CPG viewer demo.
- `demo_drl_short_train`: tiny PPO training with `STRETCH` actions and joint
  position/phase/joint-velocity observations.
- `demo_drl_target_speed_quick`: short normalized PPO target-speed run using
  `STRETCH_2`.
- `demo_drl_target_speed_full`: longer target-speed preset.
- `demo_drl_fast_swim_full`: longer fast-swimming preset.
