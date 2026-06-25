import numpy as np
import torch
import torch as th
import torch.nn as nn
from typing import Tuple

import conf
from .common import BaseExtractor, _make_mlp, _make_value_net


class AEStyleRL(nn.Module):
    """Autoencoder-style SB3 actor/critic extractor.

    Cfg key:
        No active config key is registered for this class. Legacy class name:
        ``AEStyleRL``.

    Architecture:
        Encodes the full observation into a 3-dimensional bottleneck, decodes it
        for reconstruction via ``ae_forward``, and feeds the bottleneck into
        separate actor and critic latent MLPs.

    SB3 integration:
        Returns SB3-style actor and critic latents. SB3's default ``action_net``
        is still responsible for mapping the actor latent to the Gaussian action
        mean.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
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
        """Return actor and critic latents from the encoded observation."""
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        latent = self.encoder(features)
        return self.policy_net(latent)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        latent = self.encoder(features)
        return self.value_net(latent)



class StandardConfigurableExtractor(BaseExtractor):
    """Default configurable SB3-style actor/critic extractor.

    Cfg key:
        Used when neither ``RL.localFeedback`` nor
        ``RL.stateHistoryController`` is set. Legacy class name:
        ``CustomNetwork``.

    Architecture:
        Builds separate actor and critic MLPs from ``RL.policy_network`` and
        ``RL.value_network``. The actor returns a policy latent, not an action.

    SB3 integration:
        ``latent_pi`` is passed through SB3's ``action_net`` to produce the
        Gaussian action mean. ``latent_vf`` is passed through SB3's scalar
        ``value_net``.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][-1]
        self.latent_dim_vf = conf.CONF["RL"]["value_network"]["arch"][-1]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim

        self.policy_net = _make_mlp(
            input_dim=self.feature_dim,
            hidden_dims=conf.CONF["RL"]["policy_network"]["arch"],
            activation_key="policy_network",
        )
        self.value_net = _make_value_net(self.feature_dim)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)



class PerFeatureStateHistoryExtractor(BaseExtractor):
    """State-history extractor with one temporal filter per feature.

    Cfg key:
        Set ``RL.stateHistoryController: sh1``. Legacy class name: ``sh1``.

    Architecture:
        Treats the observation as ``feature_dim / state_history_length``
        feature histories. Each feature history is compressed by its own
        learned ``Linear(state_history_length, 1)`` filter initialized as an
        average. The compressed features feed separate actor and critic MLPs.

    SB3 integration:
        Returns actor and critic latents. SB3 still applies its ``action_net``
        to the actor latent to produce action means.

    Args:
        feature_dim: Flattened observation dimension including state history.
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][-1]
        self.latent_dim_vf = conf.CONF["RL"]["value_network"]["arch"][-1]

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

        self.policy_net = _make_mlp(
            input_dim=self.feature_dim,
            hidden_dims=conf.CONF["RL"]["policy_network"]["arch"],
            activation_key="policy_network",
        )
        self.value_net = _make_value_net(self.feature_dim)

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

    def critic_features(self, features: th.Tensor) -> th.Tensor:
        return self.preprocess_state_history(features)


class GroupedStateHistoryExtractor(BaseExtractor):
    """State-history extractor with shared temporal filters per feature group.

    Cfg key:
        Set ``RL.stateHistoryController: sh2``. Legacy class name: ``sh2``.

    Architecture:
        Compresses state histories with three shared
        ``Linear(state_history_length, 1)`` filters. Features are assigned by
        ``floor(feature_index / 10)``, so the intent is to share temporal
        filters across broad groups such as positions, phases, and velocities.

    SB3 integration:
        Returns actor and critic latents. SB3 still applies its ``action_net``
        to the actor latent.

    Args:
        feature_dim: Flattened observation dimension including state history.
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][-1]
        self.latent_dim_vf = conf.CONF["RL"]["value_network"]["arch"][-1]

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

        self.policy_net = _make_mlp(
            input_dim=self.feature_dim,
            hidden_dims=conf.CONF["RL"]["policy_network"]["arch"],
            activation_key="policy_network",
        )
        self.value_net = _make_value_net(self.feature_dim)

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

    def critic_features(self, features: th.Tensor) -> th.Tensor:
        return self.preprocess_state_history(features)
