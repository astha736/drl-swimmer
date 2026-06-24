import torch
import torch as th
from torch import device
import torch.nn as nn
from typing import Callable, Tuple
import numpy as np
from gym import spaces

from stable_baselines3.common.policies import ActorCriticPolicy

import conf


# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
# class CustomNetwork(nn.Module):
#     """
#     Custom network for policy and value function.
#     It receives as input the features extracted by the features extractor.

#     :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
#     :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
#     :param last_layer_dim_vf: (int) number of units for the last layer of the value network
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#     ):
#         super().__init__()

#         # IMPORTANT:
#         # Save output dimensions, used to create the distributions
#         self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
#         self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

#         # Policy network
#         self.policy_net = nn.Sequential(
#             nn.Linear(feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
#             getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
#             nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
#             getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
#         )
#         # Value network
#         self.value_net = nn.Sequential(
#             nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
#             getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
#             nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
#             getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
#         )

#     def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
#             If all layers are shared, then ``latent_policy == latent_value``

#         Customized for very local feedback and shared net
#         """
#         return self.forward_actor(features), self.forward_critic(features)

#     def forward_actor(self, features: th.Tensor) -> th.Tensor:
#         return self.policy_net(features)

#     def forward_critic(self, features: th.Tensor) -> th.Tensor:
#         return self.value_net(features)


# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class AEStyleRL(nn.Module):
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

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, feature_dim),
        )

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
        )

    def ae_forward(self, features: th.Tensor) -> th.Tensor:
        latent = self.encoder(features)
        out = self.decoder(latent)
        return out

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        latent = self.encoder(features)
        return self.policy_net(latent)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        latent = self.encoder(features)
        return self.value_net(latent)


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
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][-1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][-1]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim

        # policy network
        layers = []
        for i in range(len(conf.CONF["RL"]["policy_network"]["arch"])):
            in_dim = (
                self.feature_dim
                if i == 0
                else conf.CONF["RL"]["policy_network"]["arch"][i - 1]
            )
            layers.append(
                nn.Linear(in_dim, conf.CONF["RL"]["policy_network"]["arch"][i])
            )
            layers.append(
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])()
            ),
        self.policy_net = nn.Sequential(*layers)

        # Value network
        layers = []
        for i in range(len(conf.CONF["RL"]["value_network"]["arch"])):
            in_dim = (
                self.feature_dim
                if i == 0
                else conf.CONF["RL"]["policy_network"]["arch"][i - 1]
            )
            layers.append(
                nn.Linear(in_dim, conf.CONF["RL"]["value_network"]["arch"][i])
            )
            layers.append(
                getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])()
            ),
        self.value_net = nn.Sequential(*layers)
        pass

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
class sh1(nn.Module):
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state_history_length = int(conf.CONF["RL"]["state_history_length"])
        self.feature_dim = int(feature_dim / conf.CONF["RL"]["state_history_length"])

        # state history filters
        def get_state_history_filter():
            return nn.Sequential(
                nn.Linear(self.state_history_length, 1),
            )

        self.state_history_filters = nn.ModuleList(
            [
                get_state_history_filter().to(self.device)
                for i in range(self.feature_dim)
            ]
        )

        # initialize weight of state history filters so that they sum up to one
        for i in range(self.feature_dim):
            torch.nn.init.constant_(
                self.state_history_filters[i][0].weight, 1 / self.state_history_length
            )
            self.state_history_filters[i][0].bias.data.fill_(0.0)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def preprocess_state_history(self, features: th.Tensor) -> th.Tensor:
        # preprocess in state history filters
        features_ = torch.tensor([], device=self.device)
        for i in range(self.feature_dim):
            idx = torch.tensor(
                [
                    i
                    for i in range(
                        i * self.state_history_length,
                        (i + 1) * self.state_history_length,
                    )
                ],
                device=self.device,
                dtype=torch.int,
            )
            out_ = self.state_history_filters[i](
                torch.index_select(features, dim=1, index=idx)
            )
            features_ = torch.cat((features_, out_), dim=1)
        return features_

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # pass to policy network
        features_ = self.preprocess_state_history(features)
        return self.policy_net(features_)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features_ = self.preprocess_state_history(features)
        return self.value_net(features_)


class sh2(nn.Module):
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state_history_length = int(conf.CONF["RL"]["state_history_length"])
        self.feature_dim = int(feature_dim / conf.CONF["RL"]["state_history_length"])

        # state history filters
        def get_state_history_filter():
            return nn.Sequential(
                nn.Linear(self.state_history_length, 1),
            )

        self.state_history_filters = nn.ModuleList(
            [get_state_history_filter().to(self.device) for i in range(3)]
        )

        # initialize weight of state history filters so that they sum up to one
        for i in range(3):
            torch.nn.init.constant_(
                self.state_history_filters[i][0].weight, 1 / self.state_history_length
            )
            self.state_history_filters[i][0].bias.data.fill_(0.0)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def preprocess_state_history(self, features: th.Tensor) -> th.Tensor:
        # preprocess in state history filters
        features_ = torch.tensor([], device=self.device)
        for i in range(self.feature_dim):
            idx = torch.tensor(
                [
                    i
                    for i in range(
                        i * self.state_history_length,
                        (i + 1) * self.state_history_length,
                    )
                ],
                device=self.device,
                dtype=torch.int,
            )
            filter_nr = int(np.floor(i / 10))
            out_ = self.state_history_filters[filter_nr](
                torch.index_select(features, dim=1, index=idx)
            )
            features_ = torch.cat((features_, out_), dim=1)
        return features_

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # pass to policy network
        features_ = self.preprocess_state_history(features)
        return self.policy_net(features_)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features_ = self.preprocess_state_history(features)
        return self.value_net(features_)


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
        action_dim: int,
    ):
        super().__init__()

        self.obs_per_iter: int = 2

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim

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

        # handle weight initialization
        # values taken from stable-baselines3
        # see https://www.google.com/search?q=why+is+ppo+using+orthogonal+initialization&oq=why+is+ppo+using+orthogonal+initialization&aqs=chrome..69i57j33i160.6735j0j1&sourceid=chrome&ie=UTF-8
        gain_last_layer = 0.01
        if "gain_last_layer" in conf.CONF["RL"]["policy_network"]:
            gain_last_layer = conf.CONF["RL"]["policy_network"]["gain_last_layer"]

        torch.nn.init.orthogonal_(self.policy_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net[4].weight, gain=gain_last_layer)
        self.policy_net[0].bias.data.fill_(0.0)
        self.policy_net[2].bias.data.fill_(0.0)
        self.policy_net[4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        idx = []
        for i in range(9):
            idx.append(
                torch.tensor([i, 11 + i], device=self.device, dtype=torch.int)
            )  # + 1 on phases,as head phase not relevant

        x0 = self.policy_net(torch.index_select(features, dim=1, index=idx[0]))
        x1 = self.policy_net(torch.index_select(features, dim=1, index=idx[1]))
        x2 = self.policy_net(torch.index_select(features, dim=1, index=idx[2]))
        x3 = self.policy_net(torch.index_select(features, dim=1, index=idx[3]))
        x4 = self.policy_net(torch.index_select(features, dim=1, index=idx[4]))
        x5 = self.policy_net(torch.index_select(features, dim=1, index=idx[5]))
        x6 = self.policy_net(torch.index_select(features, dim=1, index=idx[6]))
        x7 = self.policy_net(torch.index_select(features, dim=1, index=idx[7]))
        x8 = self.policy_net(torch.index_select(features, dim=1, index=idx[8]))

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

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

    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()

        self.obs_per_iter: int = 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim

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

        self.policy_nets = nn.ModuleList(
            [get_policy_net().to(self.device) for i in range(9)]
        )

        # handle weight initialization
        for i in range(9):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        idx = []
        for i in range(9):
            idx.append(
                torch.tensor([i, 11 + i], device=self.device, dtype=torch.int)
            )  # + 1 on phases,as head phase not relevant

        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=idx[0]))
        x1 = self.policy_nets[1](torch.index_select(features, dim=1, index=idx[1]))
        x2 = self.policy_nets[2](torch.index_select(features, dim=1, index=idx[2]))
        x3 = self.policy_nets[3](torch.index_select(features, dim=1, index=idx[3]))
        x4 = self.policy_nets[4](torch.index_select(features, dim=1, index=idx[4]))
        x5 = self.policy_nets[5](torch.index_select(features, dim=1, index=idx[5]))
        x6 = self.policy_nets[6](torch.index_select(features, dim=1, index=idx[6]))
        x7 = self.policy_nets[7](torch.index_select(features, dim=1, index=idx[7]))
        x8 = self.policy_nets[8](torch.index_select(features, dim=1, index=idx[8]))

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class nn3(nn.Module):
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
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        self.policy_net_1 = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(self.latent_dim_pi, 5),
        )

        self.policy_net_2 = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(self.latent_dim_pi, 4),
        )

        # handle weight initialization
        torch.nn.init.orthogonal_(self.policy_net_1[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_1[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_1[4].weight, gain=0.01)
        self.policy_net_1[0].bias.data.fill_(0.0)
        self.policy_net_1[2].bias.data.fill_(0.0)
        self.policy_net_1[4].bias.data.fill_(0.0)
        torch.nn.init.orthogonal_(self.policy_net_2[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_2[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_2[4].weight, gain=0.01)
        self.policy_net_2[0].bias.data.fill_(0.0)
        self.policy_net_2[2].bias.data.fill_(0.0)
        self.policy_net_2[4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        x = self.policy_net_1(features)
        y = self.policy_net_2(features)
        out = torch.cat((x, y), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class nn4(nn.Module):
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
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim = 10
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        self.policy_net_1 = nn.Sequential(
            nn.Linear(self.obs_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(self.latent_dim_pi, 5),
        )

        self.policy_net_2 = nn.Sequential(
            nn.Linear(self.obs_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(self.latent_dim_pi, 4),
        )

        # handle weight initialization
        torch.nn.init.orthogonal_(self.policy_net_1[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_1[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_1[4].weight, gain=0.01)
        self.policy_net_1[0].bias.data.fill_(0.0)
        self.policy_net_1[2].bias.data.fill_(0.0)
        self.policy_net_1[4].bias.data.fill_(0.0)
        torch.nn.init.orthogonal_(self.policy_net_2[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_2[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_2[4].weight, gain=0.01)
        self.policy_net_2[0].bias.data.fill_(0.0)
        self.policy_net_2[2].bias.data.fill_(0.0)
        self.policy_net_2[4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        idx_1 = torch.tensor(
            [0, 1, 2, 3, 4, 10, 11, 12, 13, 14], device=self.device, dtype=torch.int
        )
        idx_2 = torch.tensor(
            [5, 6, 7, 8, 9, 15, 16, 17, 18, 19], device=self.device, dtype=torch.int
        )

        x = self.policy_net_1(torch.index_select(features, dim=1, index=idx_1))
        y = self.policy_net_2(torch.index_select(features, dim=1, index=idx_2))
        out = torch.cat((x, y), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class nn5(nn.Module):
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
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim = 10
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        self.policy_net_1 = nn.Sequential(
            nn.Linear(self.obs_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(self.latent_dim_pi, 5),
        )

        # handle weight initialization
        torch.nn.init.orthogonal_(self.policy_net_1[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_1[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_1[4].weight, gain=0.01)
        self.policy_net_1[0].bias.data.fill_(0.0)
        self.policy_net_1[2].bias.data.fill_(0.0)
        self.policy_net_1[4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        idx_1 = torch.tensor(
            [0, 1, 2, 3, 4, 10, 11, 12, 13, 14], device=self.device, dtype=torch.int
        )
        idx_2 = torch.tensor(
            [5, 6, 7, 8, 9, 15, 16, 17, 18, 19], device=self.device, dtype=torch.int
        )

        x = self.policy_net_1(torch.index_select(features, dim=1, index=idx_1))
        y = self.policy_net_1(torch.index_select(features, dim=1, index=idx_2))
        out = torch.cat((x, y), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class nn6(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 6
        self.obs_dim_tail = 6
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        self.policy_net_body = nn.Sequential(
            nn.Linear(self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(self.latent_dim_pi, 1),
        )

        self.policy_net_tail = nn.Sequential(
            nn.Linear(self.obs_dim_tail, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(self.latent_dim_pi, 1),
        )

        # handle weight initialization
        torch.nn.init.orthogonal_(self.policy_net_tail[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_tail[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_tail[4].weight, gain=0.01)
        self.policy_net_tail[0].bias.data.fill_(0.0)
        self.policy_net_tail[2].bias.data.fill_(0.0)
        self.policy_net_tail[4].bias.data.fill_(0.0)

        torch.nn.init.orthogonal_(self.policy_net_body[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_body[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_body[4].weight, gain=0.01)
        self.policy_net_body[0].bias.data.fill_(0.0)
        self.policy_net_body[2].bias.data.fill_(0.0)
        self.policy_net_body[4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        # body
        idx = []
        for i in range(8):
            idx.append(
                torch.tensor(
                    [0 + i, 1 + i, 2 + i, 10 + i, 11 + i, 12 + i],
                    device=self.device,
                    dtype=torch.int,
                )
            )

        x0 = self.policy_net_body(torch.index_select(features, dim=1, index=idx[0]))
        x1 = self.policy_net_body(torch.index_select(features, dim=1, index=idx[1]))
        x2 = self.policy_net_body(torch.index_select(features, dim=1, index=idx[2]))
        x3 = self.policy_net_body(torch.index_select(features, dim=1, index=idx[3]))
        x4 = self.policy_net_body(torch.index_select(features, dim=1, index=idx[4]))
        x5 = self.policy_net_body(torch.index_select(features, dim=1, index=idx[5]))
        x6 = self.policy_net_body(torch.index_select(features, dim=1, index=idx[6]))
        x7 = self.policy_net_body(torch.index_select(features, dim=1, index=idx[7]))

        x8 = self.policy_net_tail(torch.index_select(features, dim=1, index=idx[7]))

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class nn7(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 9
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_net():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 1),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_net().to(self.device) for i in range(9)]
        )

        # handle weight initialization
        for i in range(9):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        self.idx = []
        for i in range(8):
            self.idx.append(
                torch.tensor(
                    [
                        0 + i,
                        1 + i,
                        2 + i,
                        10 + i,
                        11 + i,
                        12 + i,
                        20 + i,
                        21 + i,
                        22 + i,
                    ],
                    device=self.device,
                    dtype=torch.int,
                )
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

        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[0]))
        x1 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[1]))
        x2 = self.policy_nets[2](torch.index_select(features, dim=1, index=self.idx[2]))
        x3 = self.policy_nets[3](torch.index_select(features, dim=1, index=self.idx[3]))
        x4 = self.policy_nets[4](torch.index_select(features, dim=1, index=self.idx[4]))
        x5 = self.policy_nets[5](torch.index_select(features, dim=1, index=self.idx[5]))
        x6 = self.policy_nets[6](torch.index_select(features, dim=1, index=self.idx[6]))
        x7 = self.policy_nets[7](torch.index_select(features, dim=1, index=self.idx[7]))
        x8 = self.policy_nets[8](torch.index_select(features, dim=1, index=self.idx[7]))

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class caudl2(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 2
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_net():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 1),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_net().to(self.device) for i in range(9)]
        )

        # handle weight initialization
        for i in range(9):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        self.idx = []
        for i in range(9):
            self.idx.append(
                torch.tensor(
                    [
                        0 + i,
                        10 + i,
                    ],
                    device=self.device,
                    dtype=torch.int,
                )
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

        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[0]))
        x1 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[1]))
        x2 = self.policy_nets[2](torch.index_select(features, dim=1, index=self.idx[2]))
        x3 = self.policy_nets[3](torch.index_select(features, dim=1, index=self.idx[3]))
        x4 = self.policy_nets[4](torch.index_select(features, dim=1, index=self.idx[4]))
        x5 = self.policy_nets[5](torch.index_select(features, dim=1, index=self.idx[5]))
        x6 = self.policy_nets[6](torch.index_select(features, dim=1, index=self.idx[6]))
        x7 = self.policy_nets[7](torch.index_select(features, dim=1, index=self.idx[7]))
        x8 = self.policy_nets[8](torch.index_select(features, dim=1, index=self.idx[8]))

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class caudl(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 3
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_net():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 1),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_net().to(self.device) for i in range(9)]
        )

        # handle weight initialization
        for i in range(9):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        self.idx = []
        for i in range(9):
            self.idx.append(
                torch.tensor(
                    [
                        0 + i,
                        10 + i,
                        20 + i,
                    ],
                    device=self.device,
                    dtype=torch.int,
                )
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

        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[0]))
        x1 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[1]))
        x2 = self.policy_nets[2](torch.index_select(features, dim=1, index=self.idx[2]))
        x3 = self.policy_nets[3](torch.index_select(features, dim=1, index=self.idx[3]))
        x4 = self.policy_nets[4](torch.index_select(features, dim=1, index=self.idx[4]))
        x5 = self.policy_nets[5](torch.index_select(features, dim=1, index=self.idx[5]))
        x6 = self.policy_nets[6](torch.index_select(features, dim=1, index=self.idx[6]))
        x7 = self.policy_nets[7](torch.index_select(features, dim=1, index=self.idx[7]))
        x8 = self.policy_nets[8](torch.index_select(features, dim=1, index=self.idx[8]))

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class enn7(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 9
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_net():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 1),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_net().to(self.device) for i in range(2)]
        )

        # handle weight initialization
        for i in range(2):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        self.idx = []
        for i in range(8):
            self.idx.append(
                torch.tensor(
                    [
                        0 + i,
                        1 + i,
                        2 + i,
                        10 + i,
                        11 + i,
                        12 + i,
                        20 + i,
                        21 + i,
                        22 + i,
                    ],
                    device=self.device,
                    dtype=torch.int,
                )
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

        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[0]))
        x1 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[1]))
        x2 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[2]))
        x3 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[3]))
        x4 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[4]))
        x5 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[5]))
        x6 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[6]))
        x7 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[7]))
        x8 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[7]))

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class nn8(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 20
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_nets():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 3),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_nets().to(self.device) for i in range(3)]
        )

        # handle weight initialization
        for i in range(3):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        x0 = self.policy_nets[0](features)
        x1 = self.policy_nets[1](features)
        x2 = self.policy_nets[2](features)

        out = torch.cat((x0, x1, x2), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class nn9(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 15
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_net():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 3),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_net().to(self.device) for i in range(3)]
        )

        # handle weight initialization
        for i in range(3):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        # body
        self.idx_0 = torch.tensor(
            [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            device=self.device,
            dtype=torch.int,
        )

        self.idx_1 = torch.tensor(
            [3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 23, 24, 25, 26, 27],
            device=self.device,
            dtype=torch.int,
        )

        self.idx_2 = torch.tensor(
            [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 2, 29],
            device=self.device,
            dtype=torch.int,
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

        # pay attention on order or actions! It must go head to tail.
        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx_0))
        x1 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx_1))
        x2 = self.policy_nets[2](torch.index_select(features, dim=1, index=self.idx_2))

        out = torch.cat((x0, x1, x2), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class dnn1(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        self.policy_net_fb = nn.Sequential(
            nn.Linear(feature_dim - 4, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(self.latent_dim_pi, self.action_dim - 2),
        )

        self.policy_net_drive = nn.Sequential(
            nn.Linear(4, 16),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(16, 16),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(16, 2),
        )

        # handle weight initialization
        torch.nn.init.orthogonal_(self.policy_net_fb[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_fb[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_fb[4].weight, gain=0.01)
        self.policy_net_fb[0].bias.data.fill_(0.0)
        self.policy_net_fb[2].bias.data.fill_(0.0)
        self.policy_net_fb[4].bias.data.fill_(0.0)

        torch.nn.init.orthogonal_(self.policy_net_drive[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_drive[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_drive[4].weight, gain=0.01)
        self.policy_net_drive[0].bias.data.fill_(0.0)
        self.policy_net_drive[2].bias.data.fill_(0.0)
        self.policy_net_drive[4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        # body
        idx_0 = torch.tensor(
            [0, 1, 2, 3],
            device=self.device,
            dtype=torch.int,
        )

        idx_1 = torch.tensor(
            [i for i in range(4, features.shape[1])],
            device=self.device,
            dtype=torch.int,
        )

        # pay attention on order or actions! It must go head to tail.
        x0 = self.policy_net_drive(torch.index_select(features, dim=1, index=idx_0))
        x1 = self.policy_net_fb(torch.index_select(features, dim=1, index=idx_1))

        out = torch.cat((x0, x1), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class dnn2(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        self.policy_net_fb = nn.Sequential(
            nn.Linear(feature_dim - 4, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(self.latent_dim_pi, self.action_dim - 2),
        )

        self.policy_net_drive = nn.Sequential(
            nn.Linear(4, 16),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(16, 16),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(16, 2),
        )

        # handle weight initialization
        torch.nn.init.orthogonal_(self.policy_net_fb[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_fb[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_fb[4].weight, gain=0.01)
        self.policy_net_fb[0].bias.data.fill_(0.0)
        self.policy_net_fb[2].bias.data.fill_(0.0)
        self.policy_net_fb[4].bias.data.fill_(0.0)

        torch.nn.init.orthogonal_(self.policy_net_drive[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_drive[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_drive[4].weight, gain=0.01)
        self.policy_net_drive[0].bias.data.fill_(0.0)
        self.policy_net_drive[2].bias.data.fill_(0.0)
        self.policy_net_drive[4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        idx_1 = torch.tensor(
            [i for i in range(4, features.shape[1])],
            device=self.device,
            dtype=torch.int,
        )

        # pay attention on order or actions! It must go head to tail.
        x1 = self.policy_net_fb(torch.index_select(features, dim=1, index=idx_1))

        if conf.CONF["RL"]["curriculum"]["level"] in [2, 3, 4, 5, 6, 7]:
            if conf.CONF["RL"]["curriculum"]["current_stage"] == 0:
                # dummy value; respect correct shape
                x0 = torch.zeros(
                    [x1.shape[0], 2],
                    device=self.device,
                )
            else:
                idx_0 = torch.tensor(
                    [0, 1, 2, 3],
                    device=self.device,
                    dtype=torch.int,
                )
                x0 = self.policy_net_drive(
                    torch.index_select(features, dim=1, index=idx_0)
                )

        out = torch.cat((x0, x1), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class dnn3(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim

        self.num_filters = 2

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        # state history processing
        self.state_history_length = int(conf.CONF["RL"]["state_history_length"])
        self.feature_dim = int(feature_dim / conf.CONF["RL"]["state_history_length"])

        # state history filters
        def get_state_history_filter():
            return nn.Sequential(
                nn.Linear(self.state_history_length, 1),
            )

        self.state_history_filters = nn.ModuleList(
            [
                get_state_history_filter().to(self.device)
                for i in range(self.feature_dim * self.num_filters)
            ]
        )

        # initialize weight of state history filters so that they sum up to one
        for i in range(self.feature_dim * self.num_filters):
            torch.nn.init.constant_(
                self.state_history_filters[i][0].weight, 1 / self.state_history_length
            )
            self.state_history_filters[i][0].bias.data.fill_(0.0)

        self.policy_net_fb = nn.Sequential(
            nn.Linear(
                (self.feature_dim - 4) * self.num_filters,
                conf.CONF["RL"]["policy_network"]["arch"][0],
            ),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(self.latent_dim_pi, self.action_dim - 2),
        )

        self.policy_net_drive = nn.Sequential(
            nn.Linear(4 * self.num_filters, 16),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(16, 16),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(16, 2),
        )

        # handle weight initialization
        torch.nn.init.orthogonal_(self.policy_net_fb[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_fb[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_fb[4].weight, gain=0.01)
        self.policy_net_fb[0].bias.data.fill_(0.0)
        self.policy_net_fb[2].bias.data.fill_(0.0)
        self.policy_net_fb[4].bias.data.fill_(0.0)

        torch.nn.init.orthogonal_(self.policy_net_drive[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_drive[2].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.policy_net_drive[4].weight, gain=0.01)
        self.policy_net_drive[0].bias.data.fill_(0.0)
        self.policy_net_drive[2].bias.data.fill_(0.0)
        self.policy_net_drive[4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(
                self.feature_dim * self.num_filters,
                conf.CONF["RL"]["value_network"]["arch"][0],
            ),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        self.idx_0 = torch.tensor(
            [i for i in range(4 * self.num_filters)],
            device=self.device,
            dtype=torch.int,
        )

        self.idx_1 = torch.tensor(
            [
                i
                for i in range(
                    4 * self.num_filters, self.feature_dim * self.num_filters
                )
            ],
            device=self.device,
            dtype=torch.int,
        )

        self.idxs = []
        for i in range(self.feature_dim):
            self.idxs.append(
                torch.tensor(
                    [
                        i
                        for i in range(
                            i * self.state_history_length,
                            (i + 1) * self.state_history_length,
                        )
                    ],
                    device=self.device,
                    dtype=torch.int,
                )
            )

    def preprocess_state_history(self, features: th.Tensor) -> th.Tensor:
        # preprocess in state history filters
        # also process target_vel, current_vel via state history
        features_ = torch.tensor([], device=self.device)
        for i in range(self.feature_dim):
            for j in range(self.num_filters):
                out_ = self.state_history_filters[self.num_filters * i + j](
                    torch.index_select(features, dim=1, index=self.idxs[i])
                )
                features_ = torch.cat((features_, out_), dim=1)
        return features_

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases
        features_ = self.preprocess_state_history(features)

        # pay attention on order or actions! It must go head to tail.
        x1 = self.policy_net_fb(torch.index_select(features_, dim=1, index=self.idx_1))

        if conf.CONF["RL"]["curriculum"]["level"] in [2, 3, 4, 5, 6, 7]:
            if conf.CONF["RL"]["curriculum"]["current_stage"] == 0:
                # dummy value; respect correct shape
                x0 = torch.zeros(
                    [x1.shape[0], 2],
                    device=self.device,
                )
            else:
                x0 = self.policy_net_drive(
                    torch.index_select(features_, dim=1, index=self.idx_0)
                )

        out = torch.cat((x0, x1), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features_ = self.preprocess_state_history(features)

        return self.value_net(features_)


class enn8(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 15
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_net():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 3),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_net().to(self.device) for i in range(2)]
        )

        # handle weight initialization
        for i in range(2):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        # body
        self.idx_0 = torch.tensor(
            [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            device=self.device,
            dtype=torch.int,
        )

        self.idx_1 = torch.tensor(
            [3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 23, 24, 25, 26, 27],
            device=self.device,
            dtype=torch.int,
        )

        self.idx_2 = torch.tensor(
            [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 2, 29],
            device=self.device,
            dtype=torch.int,
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

        # pay attention on order or actions! It must go head to tail.
        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx_0))
        x1 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx_1))
        x2 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx_2))

        out = torch.cat((x0, x1, x2), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class enn1(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 30
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_nets():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 1),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_nets().to(self.device) for i in range(9)]
        )

        # handle weight initialization
        for i in range(9):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases; 20-29 joint vels

        x0 = self.policy_nets[0](features)
        x1 = self.policy_nets[1](features)
        x2 = self.policy_nets[2](features)
        x3 = self.policy_nets[3](features)
        x4 = self.policy_nets[4](features)
        x5 = self.policy_nets[5](features)
        x6 = self.policy_nets[6](features)
        x7 = self.policy_nets[7](features)
        x8 = self.policy_nets[8](features)

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class enn2(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 30
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_nets():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 3),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_nets().to(self.device) for i in range(3)]
        )

        # handle weight initialization
        for i in range(3):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases; 20-29 joint vels

        x0 = self.policy_nets[0](features)
        x1 = self.policy_nets[1](features)
        x2 = self.policy_nets[2](features)

        out = torch.cat((x0, x1, x2), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class enn3(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 21
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_nets():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 1),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_nets().to(self.device) for i in range(9)]
        )

        # handle weight initialization
        for i in range(9):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        self.idx_0 = torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
            ],
            device=self.device,
            dtype=torch.int,
        )
        self.idx_1 = torch.tensor(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
            ],
            device=self.device,
            dtype=torch.int,
        )
        self.idx_2 = torch.tensor(
            [
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
            ],
            device=self.device,
            dtype=torch.int,
        )
        self.idx_3 = torch.tensor(
            [
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
            ],
            device=self.device,
            dtype=torch.int,
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases; 20-29 joint vels

        # pay attention on order or actions! It must go head to tail.
        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx_0))
        x1 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx_0))
        x2 = self.policy_nets[2](torch.index_select(features, dim=1, index=self.idx_0))
        x3 = self.policy_nets[3](torch.index_select(features, dim=1, index=self.idx_1))
        x4 = self.policy_nets[4](torch.index_select(features, dim=1, index=self.idx_2))
        x5 = self.policy_nets[5](torch.index_select(features, dim=1, index=self.idx_3))
        x6 = self.policy_nets[6](torch.index_select(features, dim=1, index=self.idx_3))
        x7 = self.policy_nets[7](torch.index_select(features, dim=1, index=self.idx_3))
        x8 = self.policy_nets[8](torch.index_select(features, dim=1, index=self.idx_3))

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class enn4(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 15
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_nets():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 1),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_nets().to(self.device) for i in range(9)]
        )

        # handle weight initialization
        for i in range(9):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        self.idx = []
        for i in range(6):
            self.idx.append(
                torch.tensor(
                    [
                        0 + i,
                        1 + i,
                        2 + i,
                        3 + i,
                        4 + i,
                        10 + i,
                        11 + i,
                        12 + i,
                        13 + i,
                        14 + i,
                        20 + i,
                        21 + i,
                        22 + i,
                        23 + i,
                        24 + i,
                    ],
                    device=self.device,
                    dtype=torch.int,
                )
            )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases; 20-29 joint vels

        # pay attention on order or actions! It must go head to tail.
        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[0]))
        x1 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[0]))
        x2 = self.policy_nets[2](torch.index_select(features, dim=1, index=self.idx[1]))
        x3 = self.policy_nets[3](torch.index_select(features, dim=1, index=self.idx[2]))
        x4 = self.policy_nets[4](torch.index_select(features, dim=1, index=self.idx[3]))
        x5 = self.policy_nets[5](torch.index_select(features, dim=1, index=self.idx[4]))
        x6 = self.policy_nets[6](torch.index_select(features, dim=1, index=self.idx[5]))
        x7 = self.policy_nets[7](torch.index_select(features, dim=1, index=self.idx[5]))
        x8 = self.policy_nets[8](torch.index_select(features, dim=1, index=self.idx[5]))

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class enn6(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 15
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_nets():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 1),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_nets().to(self.device) for i in range(4)]
        )

        # handle weight initialization
        for i in range(4):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        self.idx = []
        for i in range(6):
            self.idx.append(
                torch.tensor(
                    [
                        0 + i,
                        1 + i,
                        2 + i,
                        3 + i,
                        4 + i,
                        10 + i,
                        11 + i,
                        12 + i,
                        13 + i,
                        14 + i,
                        20 + i,
                        21 + i,
                        22 + i,
                        23 + i,
                        24 + i,
                    ],
                    device=self.device,
                    dtype=torch.int,
                )
            )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases; 20-29 joint vels

        # pay attention on order or actions! It must go head to tail.
        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx[0]))
        x1 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[0]))
        x2 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[1]))
        x3 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[2]))
        x4 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[3]))
        x5 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[4]))
        x6 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx[5]))
        x7 = self.policy_nets[2](torch.index_select(features, dim=1, index=self.idx[5]))
        x8 = self.policy_nets[3](torch.index_select(features, dim=1, index=self.idx[5]))

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class enn5(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_dim_body = 21
        self.action_dim = action_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        def get_policy_net():
            return nn.Sequential(
                nn.Linear(
                    self.obs_dim_body, conf.CONF["RL"]["policy_network"]["arch"][0]
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(
                    conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi
                ),
                getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
                nn.Linear(self.latent_dim_pi, 3),
            )

        self.policy_nets = nn.ModuleList(
            [get_policy_net().to(self.device) for i in range(3)]
        )

        # handle weight initialization
        for i in range(3):
            torch.nn.init.orthogonal_(self.policy_nets[i][0].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][2].weight, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(self.policy_nets[i][4].weight, gain=0.01)
            self.policy_nets[i][0].bias.data.fill_(0.0)
            self.policy_nets[i][2].bias.data.fill_(0.0)
            self.policy_nets[i][4].bias.data.fill_(0.0)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["value_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["value_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["value_network"]["act_fn"])(),
        )

        torch.nn.init.orthogonal_(self.value_net[0].weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_net[2].weight, gain=np.sqrt(2))
        self.value_net[0].bias.data.fill_(0.0)
        self.value_net[2].bias.data.fill_(0.0)

        # body
        self.idx_0 = torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
            ],
            device=self.device,
            dtype=torch.int,
        )
        self.idx_1 = torch.tensor(
            [
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
            ],
            device=self.device,
            dtype=torch.int,
        )
        self.idx_2 = torch.tensor(
            [
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
            ],
            device=self.device,
            dtype=torch.int,
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

        # pay attention on order or actions! It must go head to tail.
        x0 = self.policy_nets[0](torch.index_select(features, dim=1, index=self.idx_0))
        x1 = self.policy_nets[1](torch.index_select(features, dim=1, index=self.idx_1))
        x2 = self.policy_nets[2](torch.index_select(features, dim=1, index=self.idx_2))

        out = torch.cat((x0, x1, x2), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

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
        self.ortho_init = True

    def _build_mlp_extractor(self, action_dim: int = None) -> None:
        if action_dim is None:
            action_dim = int(np.prod(self.action_space.shape))

        network_registry = {
            "shared": localFeedbackShared,
            "non-shared": localFeedbackNonShared,
            "nn3": nn3,
            "nn4": nn4,
            "nn6": nn6,
            "nn7": nn7,
            "caudl": caudl,
            "caudl2": caudl2,
            "nn8": nn8,
            "nn9": nn9,
            "enn1": enn1,
            "enn2": enn2,
            "enn3": enn3,
            "enn4": enn4,
            "enn5": enn5,
            "enn6": enn6,
            "enn7": enn7,
            "enn8": enn8,
            "dnn1": dnn1,
            "dnn2": dnn2,
            "dnn3": dnn3,
        }
        state_history_registry = {
            "sh1": sh1,
            "sh2": sh2,
        }

        local_feedback = conf.CONF["RL"].get("localFeedback")
        state_history_controller = conf.CONF["RL"].get("stateHistoryController")

        if local_feedback or state_history_controller:
            if local_feedback == "nn5":
                raise NotImplementedError
            if local_feedback in network_registry:
                self.mlp_extractor = network_registry[local_feedback](
                    self.features_dim, action_dim
                )
            elif state_history_controller in state_history_registry:
                self.mlp_extractor = state_history_registry[state_history_controller](
                    self.features_dim
                )
            else:
                raise NotImplementedError(
                    f"Unknown localFeedback/stateHistoryController: "
                    f"{local_feedback}/{state_history_controller}"
                )

        else:
            self.mlp_extractor = CustomNetwork(self.features_dim)

        if hasattr(self.mlp_extractor, "policy_net"):
            conf.CONF["misc"]["log_num_trainable_params"] = sum(
                p.numel()
                for p in self.mlp_extractor.policy_net.parameters()
                if p.requires_grad
            )
        elif hasattr(self.mlp_extractor, "policy_nets"):
            conf.CONF["misc"]["log_num_trainable_params"] = sum(
                p.numel()
                for p in self.mlp_extractor.policy_nets.parameters()
                if p.requires_grad
            )
        elif hasattr(self.mlp_extractor, "policy_net_drive") and hasattr(
            self.mlp_extractor, "policy_net_fb"
        ):
            drive_params = sum(
                p.numel()
                for p in self.mlp_extractor.policy_net_drive.parameters()
                if p.requires_grad
            )
            fb_params = sum(
                p.numel()
                for p in self.mlp_extractor.policy_net_fb.parameters()
                if p.requires_grad
            )
            conf.CONF["misc"][
                "log_num_trainable_params"
            ] = f"policy_net_drive: {drive_params}, policy_net_fb: {fb_params}"
        else:
            raise AttributeError(
                "Could not find a policy network on the selected extractor."
            )
