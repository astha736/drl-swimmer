import torch
import torch as th
from torch import device, nn
from typing import Callable, Tuple

from stable_baselines3.common.policies import ActorCriticPolicy
from gym import spaces

import conf


# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # Thats wrong. Action space is one dimensional! (Same probability distributions for all actions)
        # Make action space 1D
        # not sure how to handle obs space yet
        # feature = th.Tensor([th.Tensor([features[0][1]])])
        # out = self.policy_net(feature)
        # return th.Tensor([out, out, out, out, out, out, out, out, out, out])

        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class CustomNetwork2(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = 1
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(2, 20),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(20, 1),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # Thats wrong. Action space is one dimensional! (Same probability distributions for all actions)
        # Make action space 1D
        # not sure how to handle obs space yet

        # 0-9: joint positions
        # 10-19: phases
        out = torch.stack(
            [
                self.policy_net(
                    th.tensor(
                        [
                            th.Tensor([features[0][i]]),
                            th.Tensor([features[0][i + 10 + 1]]),
                        ],
                        device="cuda:0" if features.get_device() == 0 else "cpu",
                    )
                )
                for i in range(9)
            ]
        )
        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class CustomNetwork3(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()

        self.n_calls: int = 0
        self.obs_per_iter: int = 2

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.obs_per_iter, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # extract relevant obs
        # 0-9: joint positions
        # 10-19: phases
        obs = th.tensor(
            [
                th.Tensor([features[0][self.n_calls]]),
                th.Tensor(
                    [features[0][self.n_calls + 1 + 10]]
                ),  # + 1 because head phase not relevant
            ],
            device="cuda:0" if features.get_device() == 0 else "cpu",
        )

        # increment call counter and reset when = 10
        # don't try to make this as one-lines ;)
        self.n_calls += 1
        if self.n_calls == 9:
            self.n_calls = 0

        return self.policy_net(obs)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        # self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork3(self.features_dim)
