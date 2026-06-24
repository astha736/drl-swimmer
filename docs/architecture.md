# Architecture

The pipeline is organized around named experiment configurations. A single run
selects one YAML entry, builds the FARMS/MuJoCo simulation, applies the
AgnathaX oscillator architecture, and then either runs a fixed architecture test
or trains/evaluates a Stable-Baselines policy.

## Runtime Flow

1. `main.py` parses `--experiment_config`, `--experiment_id`, optional
   `--base_test_path`, `--date`, `--seed`, and `--log_level`.
2. `conf.py` loads the selected YAML entry, expands defaults, loads referenced
   PPO/SAC/reward YAML files, and creates log paths.
3. `farms_sim.simulation.setup_from_clargs` loads the animat, arena, and
   simulation YAML files from `config/_ANIMAT`, `config/_ARENA`, and
   `config/_SIMULATION`.
4. `utils/limbless_experiment_options.py` creates a `RobotInitialConditions`
   object and applies network, domain, spawn, and oscillator initialization.
5. `utils/simulation.py` creates FARMS data, the AgnathaX ODE network, the
   FARMS controller, callbacks, and the MuJoCo simulation object.
6. `utils/train_test.py` dispatches to architecture testing, training, or saved
   model testing.

## Main Components

- `agnathax_control/`: Cython-backed ODE and network implementation used by the
  FARMS controller.
- `utils/limbless_network.py`: builds oscillator-to-oscillator coupling and
  joint/contact-to-oscillator feedback maps.
- `utils/limbless_spawn.py`: sets initial body shape, pose, joint state, link
  velocities, and COM velocity.
- `utils/limbless_oscillator.py`: sets oscillator initial phase/amplitude
  conditions.
- `utils/limbless_domain.py`: applies domain parameters such as friction.
- `rlgym/rl_gym.py`: provides the Gym environment, action/observation spaces,
  reward computation, episode reset randomization, early termination, and
  evaluation callbacks.
- `utils/networks.py`: custom Stable-Baselines actor/critic feature networks,
  including plain MLPs, autoencoder-style networks, and state-history variants.
- `utils/train_test.py`: creates vectorized environments, configures PPO/SAC,
  handles VecNormalize, saves checkpoints, evaluates policies, and aggregates
  cross-seed metrics.

## Controller Architecture

The robot model uses 10 body joints and left/right oscillators per body segment.
`RobotInitialNetwork` can create:

- intra-segmental left/right oscillator coupling,
- inter-segmental nearest-neighbor oscillator coupling,
- local stretch feedback from a joint to the same segment,
- caudal stretch feedback from a joint to the next caudal oscillator,
- rostral stretch feedback from a joint to the next rostral oscillator,
- reaction-force feedback through contact-to-oscillator maps.

Feedback sensitivity names in experiment YAML map to FARMS connection types:

- `COS`: `STRETCH2AMPTEGOTAE`
- `SIN`: `STRETCH2FREQTEGOTAE`
- `NONPERIOD`: `STRETCH2FREQ`
- `SIGNONE`: `STRETCH2AMP`

For the current RL setup, `conf.py` requires exactly one of
`robot_arch.s_caudl_weight` or `robot_arch.s_local_weight` to be non-null.

## Training And Evaluation

Training uses `make_vec_env` with one environment by default. If
`RL.normWrapper` is true, the environment is wrapped in `VecNormalize` for
observation and optional reward normalization.

PPO uses `utils.networks.CustomActorCriticPolicy`, which constructs policy and
value networks from `RL.policy_network` and `RL.value_network`. If
`value_network` is omitted, it defaults to the policy-network shape.

During training, `EvalCallback` periodically evaluates and writes
`best_model.zip`. At the end of training, `last_model_trained.zip` is always
saved; short runs fall back to saving the final model as `best_model.zip` if no
evaluation checkpoint was created.

Testing loads `best_model.zip`, restores `best_model_normalize.pkl` when needed,
runs fixed-seed evaluation, writes `eval_metrics.yaml`, and can produce
single-episode plots/videos. Cross-seed evaluation is controlled by
`evaluation.cross_seed_eval`.

## Logs And Artifacts

Runs write under:

```text
experiments/<experiment_id>/logs/<date>/<seed>/
```

Common artifacts:

- `best_model.zip`
- `last_model_trained.zip`
- `best_model_normalize.pkl` and `last_model_trained_normalize.pkl`
- TensorBoard event files
- `eval_metrics.yaml`
- `single_test_env_metrics.txt`
- PDF plots and optional videos
- `simulation.profile`

On machines with `/shared`, TensorBoard and observation buffers may be redirected
to `/shared/hausdoer/...`; otherwise logs stay in the local experiment folder.
