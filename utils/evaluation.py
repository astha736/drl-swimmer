import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)


def _read_env_metrics(env: VecEnv) -> Dict[str, Any]:
    try:
        return env.venv.envs[0].env.metrics
    except AttributeError:
        return env.envs[0].env.metrics


def _append_action_gradients(model: "type_aliases.PolicyPredictor", observations: np.ndarray) -> None:
    import conf

    if not isinstance(conf.CONF.get("misc", {}).get("log_grads"), list):
        return

    observation_tensor, _ = model.policy.obs_to_tensor(observations)
    observation_tensor.requires_grad = True
    grads = []

    for action_idx in range(model.policy.action_space.shape[0]):
        if observation_tensor.grad is not None:
            observation_tensor.grad.zero_()
        actions = model.policy._predict(observation_tensor, deterministic=True)
        actions[0][action_idx].backward()
        grads.append(observation_tensor.grad.cpu().numpy().copy())

    conf.CONF["misc"]["log_grads"].append(grads.copy())


def evaluate_policy_with_metrics(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    custom_metrics: bool = False,
    log_gradients: bool = False,
) -> Union[Tuple[float, float, Dict[str, Any]], Tuple[List[float], List[int], Dict[str, Any]]]:
    is_monitor_wrapped = False

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    metrics = {}

    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        if log_gradients:
            _append_action_gradients(model, observations)

        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] >= episode_count_targets[i]:
                continue

            reward = rewards[i]
            done = dones[i]
            info = infos[i]
            episode_starts[i] = done

            if callback is not None:
                callback(locals(), globals())

            if dones[i]:
                if is_monitor_wrapped:
                    if "episode" not in info:
                        continue
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
                else:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])

                episode_counts[i] += 1
                current_rewards[i] = 0
                current_lengths[i] = 0

                if custom_metrics:
                    env_metrics = _read_env_metrics(env)
                    for metric, value in env_metrics.items():
                        metrics.setdefault(metric, []).append(value)

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    for metric, values in metrics.items():
        metrics[metric] = f"[{np.mean(values):.4f}, {np.std(values):.4f}]"

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )

    if return_episode_rewards:
        return episode_rewards, episode_lengths, metrics
    return mean_reward, std_reward, metrics
