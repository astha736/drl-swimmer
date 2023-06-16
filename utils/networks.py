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
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class localFeedbackShared(nn.Module):
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
            nn.Linear(
                self.latent_dim_pi, 1
            ),  # 1 output neuron for each action; replaces the proba_distribution_net of stable-baselines3
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        mean_actions = th.zeros(
            [1, 9],
            device="cuda:0" if features.get_device() == 0 else "cpu",
        )

        for i in range(9):
            feature = th.tensor(
                [
                    th.Tensor([features[0][i]]),
                    th.Tensor(
                        [features[0][i + 1 + 10]]
                    ),  # + 1 because head phase not relevant
                ],
                device="cuda:0" if features.get_device() == 0 else "cpu",
            )
            mean_actions[0][i] = self.policy_net(feature)

        return mean_actions

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class localFeedbackNonShared(nn.Module):
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

        self.obs_per_iter: int = 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_net():
            return nn.Sequential(
                nn.Linear(
                    self.obs_per_iter, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    self.latent_dim_pi, 1
                ),  # 1 output neuron for each action; replaces the proba_distribution_net of stable-baselines3
            )

        self.policy_nets = [get_policy_net().to(self.device) for i in range(9)]

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        mean_actions = th.zeros(
            [1, 9],
            device="cuda:0" if features.get_device() == 0 else "cpu",
        )

        for i in range(9):
            feature = th.tensor(
                [
                    th.Tensor([features[0][i]]),
                    th.Tensor(
                        [features[0][i + 1 + 10]]
                    ),  # + 1 because head phase not relevant
                ],
                device="cuda:0" if features.get_device() == 0 else "cpu",
            )
            mean_actions[0][i] = self.policy_nets[i](feature)

        return mean_actions

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
        # choose correct network
        if conf.CONF["RL"]["localFeedback"]:
            if conf.CONF["RL"]["localFeedback"] == "shared":
                self.mlp_extractor = localFeedbackShared(self.features_dim)
            elif conf.CONF["RL"]["localFeedback"] == "non-shared":
                self.mlp_extractor = localFeedbackNonShared(self.features_dim)
        else:
            self.mlp_extractor = CustomNetwork(self.features_dim)
