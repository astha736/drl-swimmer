# Experiment Configs

Named experiments are YAML entries. Run one with:

```bash
python main.py -c config/_EXPERIMENT/demo.yaml -e demo_arch_test -d demo -s 999
```

Important fields:

- `run_type`: `arch_testing` for a CPG/feedback simulation, or `train` for RL.
- `robot_arch`: CPG and feedback initialization. Exactly one of `s_caudl_weight` and `s_local_weight` should be non-null.
- `config.animat`: robot model/control config.
- `config.arena`: arena and water config.
- `config.simulation`: MuJoCo/FARMS runtime config. Use `headless: true` on servers.
- `RL.action_choice`: policy outputs, such as `STRETCH` or `DRIVE`.
- `RL.observation_choice`: policy inputs, such as `VELOCITIES`, `JOINT_POSITION`, `PHASES`, `JOINT_VEL`.
- `RL.RewardFnc`: path to reward weights.
- `RL.PPOparams`: path to Stable-Baselines PPO hyperparameters.
- `training.eval_freq`: optional short-demo override for evaluation frequency.
- `evaluation.n_eval_episodes`: optional override for evaluation length.
- `evaluation.cross_seed_eval`: set `false` for demos.
