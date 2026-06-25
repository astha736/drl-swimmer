import torch
import torch as th
import torch.nn as nn
from typing import Optional, Sequence, Tuple

import conf


def _activation_from_conf(network_key: str) -> nn.Module:
    return getattr(torch.nn, conf.CONF["RL"][network_key]["act_fn"])()


def _make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation_key: str,
    output_dim: Optional[int] = None,
) -> nn.Sequential:
    """Build a simple linear MLP from the active RL config.

    The hidden layers use ``conf.CONF["RL"][activation_key]["act_fn"]`` after
    each linear layer. ``output_dim`` is optional because SB3-style extractors
    return latent vectors, while action-mean extractors often add their own
    final action-output layer.
    """
    layers = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(_activation_from_conf(activation_key))
        in_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def _make_value_net(input_dim: int) -> nn.Sequential:
    """Build the critic latent network from ``RL.value_network``.

    ``conf.init`` defaults ``RL.value_network`` to ``RL.policy_network`` when no
    separate critic architecture is configured, so this matches the existing
    experiments while allowing future actor/critic architectures to differ.
    """
    return _make_mlp(
        input_dim=input_dim,
        hidden_dims=conf.CONF["RL"]["value_network"]["arch"],
        activation_key="value_network",
    )


class BaseExtractor(nn.Module):
    """Base class for SB3 ``mlp_extractor`` replacements.

    SB3's ``ActorCriticPolicy`` expects an extractor whose ``forward`` method
    returns ``(latent_pi, latent_vf)``. ``latent_pi`` feeds the policy/action
    path and ``latent_vf`` feeds SB3's scalar ``value_net``. Subclasses must
    implement ``forward_actor`` and may override ``critic_features`` when the
    critic needs preprocessing such as state-history compression.
    """

    outputs_action_mean = False

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Return the actor latent or final action mean for this extractor."""
        raise NotImplementedError

    def critic_features(self, features: th.Tensor) -> th.Tensor:
        return features

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self.critic_features(features))


class ActionMeanExtractor(BaseExtractor):
    """Base class for extractors whose actor branch returns action means.

    SB3 normally maps ``latent_pi`` through ``policy.action_net``. Classes that
    inherit from this base already include their final action-output layer, so
    ``CustomActorCriticPolicy`` bypasses ``action_net`` and passes the returned
    tensor directly to the Gaussian action distribution.
    """

    outputs_action_mean = True
