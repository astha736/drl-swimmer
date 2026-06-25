from typing import Callable

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy

import conf
from .extractors import StandardConfigurableExtractor
from .registries import (
    NETWORK_ALIASES,
    NETWORK_REGISTRY,
    STATE_HISTORY_REGISTRY,
    _valid_local_feedback_message,
)


class CustomActorCriticPolicy(ActorCriticPolicy):
    """SB3 actor-critic policy that selects the configured custom extractor.

    Cfg key:
        Use this class as the SB3 policy. The selected extractor is controlled by
        ``RL.localFeedback`` for action-mean heads or
        ``RL.stateHistoryController`` for state-history extractors. If neither
        key is set, ``StandardConfigurableExtractor`` is used. ``RL.localFeedback``
        accepts both the old registry names and the readable names in
        ``NETWORK_ALIASES``.

    Architecture:
        Replaces SB3's default ``mlp_extractor`` with one of the registered
        custom classes while keeping SB3's optimizer, log standard deviation,
        value head, and distribution machinery.

    SB3 integration:
        For standard extractors, ``latent_pi`` still passes through SB3's
        ``action_net``. For ``ActionMeanExtractor`` subclasses,
        ``latent_pi`` is already the Gaussian action mean, so this policy
        bypasses ``action_net`` without patching SB3 globally.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        uses_local_feedback = bool(conf.CONF["RL"].get("localFeedback"))
        if uses_local_feedback:
            # Local-feedback networks initialize their own final action heads.
            # Disable SB3's recursive orthogonal init so those gains are kept.
            kwargs["ortho_init"] = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        if uses_local_feedback:
            # Preserve SB3's initialization for the scalar value head while leaving
            # the custom local-feedback action heads untouched.
            self.value_net.apply(self.init_weights)

    def _build_mlp_extractor(self, action_dim: int = None) -> None:
        if action_dim is None:
            action_dim = int(np.prod(self.action_space.shape))

        network_name_raw = conf.CONF["RL"].get("localFeedback")
        network_name = NETWORK_ALIASES.get(network_name_raw, network_name_raw)
        history_name = conf.CONF["RL"].get("stateHistoryController")

        if network_name:
            if network_name == "nn5":
                raise NotImplementedError(
                    f"localFeedback '{network_name_raw}' resolves to old key "
                    "'nn5', which is registered but not implemented for the "
                    "current 9-action swimmer output shape"
                )
            try:
                extractor_cls = NETWORK_REGISTRY[network_name]
            except KeyError as exc:
                raise NotImplementedError(
                    f"Unknown localFeedback network '{network_name_raw}'. "
                    f"{_valid_local_feedback_message()}"
                ) from exc
            self.mlp_extractor = extractor_cls(self.features_dim, action_dim)
        elif history_name:
            try:
                extractor_cls = STATE_HISTORY_REGISTRY[history_name]
            except KeyError as exc:
                raise NotImplementedError(
                    f"Unknown stateHistoryController '{history_name}'"
                ) from exc
            self.mlp_extractor = extractor_cls(self.features_dim)
        else:
            self.mlp_extractor = StandardConfigurableExtractor(self.features_dim)

        try:
            conf.CONF["misc"]["log_num_trainable_params"] = sum(
                p.numel()
                for p in self.mlp_extractor.policy_net.parameters()
                if p.requires_grad
            )
        except:
            try:
                conf.CONF["misc"]["log_num_trainable_params"] = sum(
                    p.numel()
                    for p in self.mlp_extractor.policy_nets.parameters()
                    if p.requires_grad
                )
            except:
                conf.CONF["misc"][
                    "log_num_trainable_params"
                ] = f"policy_net_drive: {sum(p.numel()for p in self.mlp_extractor.policy_net_drive.parameters()if p.requires_grad)}, policy_net_fb: {sum(p.numel() for p in self.mlp_extractor.policy_net_fb.parameters() if p.requires_grad)}"

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        if getattr(self.mlp_extractor, "outputs_action_mean", False):
            mean_actions = latent_pi
        else:
            mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        if isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        if isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        if isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        raise ValueError("Invalid action distribution")
