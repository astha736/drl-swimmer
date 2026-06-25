import torch
import torch as th
from torch import device
import torch.nn as nn
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from gym import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)

import conf


class ObservationLayout:
    """Index helper for robot observation vectors built by ``ObservationChoice``.

    The Gym observation is a concatenation of the configured
    ``RL.observation_choice`` blocks. This class turns semantic requests such as
    "joint positions 3-7 plus phases 3-7" into the actual column indices for the
    current configuration. It keeps AgnathaX/FARMS observation layout knowledge
    out of the network architecture code.
    """

    JOINT_SIZED_BLOCKS = {
        "JOINT_POSITION",
        "JOINT_VEL",
        "PHASES",
        "AMPLITUDES",
        "PHASE_DIFF_REL",
        "PHASE_DIFF_ABS",
    }

    CONTACT_SIZED_BLOCKS = {
        "REACTION_X",
        "REACTION_Y",
        "REACTION_Z",
        "REACTION_XY",
        "REACTION_XYZ",
    }

    FIXED_BLOCK_LENGTHS = {
        "VELOCITIES": 4,
    }

    FIELD_TO_BLOCK = {
        "position": "JOINT_POSITION",
        "phase": "PHASES",
        "velocity": "JOINT_VEL",
        "body_velocity": "VELOCITIES",
    }

    def __init__(
        self,
        observation_list: Sequence[object],
        n_body_joints: int = 10,
    ):
        self.observation_names = [self._observation_name(obs) for obs in observation_list]
        self.n_body_joints = n_body_joints
        self.offsets: Dict[str, int] = {}

        offset = 0
        for name in self.observation_names:
            self.offsets[name] = offset
            offset += self._block_length(name)
        self.size = offset

    @classmethod
    def from_conf(cls) -> "ObservationLayout":
        return cls(conf.CONF["RL"]["observation_choice"])

    @staticmethod
    def _observation_name(observation: object) -> str:
        if hasattr(observation, "name"):
            return observation.name
        return str(observation)

    def _block_length(self, block_name: str) -> int:
        if block_name in self.JOINT_SIZED_BLOCKS:
            return self.n_body_joints
        if block_name in self.CONTACT_SIZED_BLOCKS:
            return self.n_body_joints + 1
        try:
            return self.FIXED_BLOCK_LENGTHS[block_name]
        except KeyError as exc:
            raise ValueError(f"Unknown observation block '{block_name}'") from exc

    def block(self, block_name: str) -> Tuple[int, ...]:
        try:
            start = self.offsets[block_name]
        except KeyError as exc:
            raise ValueError(
                f"Observation block '{block_name}' is required by this network "
                f"but is not present in RL.observation_choice={self.observation_names}"
            ) from exc
        return tuple(range(start, start + self._block_length(block_name)))

    def field(self, field_name: str) -> Tuple[int, ...]:
        return self.block(self.FIELD_TO_BLOCK[field_name])

    def slice(
        self,
        field_name: str,
        start_joint: int,
        width: int,
    ) -> Tuple[int, ...]:
        block_name = self.FIELD_TO_BLOCK[field_name]
        block_start = self.offsets[block_name]
        return tuple(range(block_start + start_joint, block_start + start_joint + width))

    def window(
        self,
        start_joint: int,
        width: int,
        fields: Sequence[str] = ("position", "phase", "velocity"),
    ) -> Tuple[int, ...]:
        indices: List[int] = []
        for field_name in fields:
            indices.extend(self.slice(field_name, start_joint, width))
        return tuple(indices)

    def single_joint(
        self,
        joint_idx: int,
        fields: Sequence[str] = ("position", "phase"),
    ) -> Tuple[int, ...]:
        return self.window(joint_idx, 1, fields)

    def without(self, excluded_indices: Iterable[int]) -> Tuple[int, ...]:
        return self.without_from_size(self.size, excluded_indices)

    @staticmethod
    def without_from_size(
        total_size: int,
        excluded_indices: Iterable[int],
    ) -> Tuple[int, ...]:
        excluded = set(excluded_indices)
        return tuple(idx for idx in range(total_size) if idx not in excluded)

    def history_indices(
        self,
        base_indices: Iterable[int],
        num_filters: int,
    ) -> Tuple[int, ...]:
        indices: List[int] = []
        for base_idx in base_indices:
            start = base_idx * num_filters
            indices.extend(range(start, start + num_filters))
        return tuple(indices)

    @staticmethod
    def tensor(indices: Iterable[int], device: torch.device) -> th.Tensor:
        return torch.tensor(tuple(indices), device=device, dtype=torch.long)


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



class ObsLocalActJointSharedActionHead(ActionMeanExtractor):
    """Shared local-feedback action-mean head.

    Cfg key:
        Set ``RL.localFeedback: shared``. Legacy class name:
        ``localFeedbackShared``.

    Architecture:
        Reuses one small MLP for all action outputs. For each of the 9 action
        positions, the actor selects a local pair of joint position and phase
        features, maps that pair to one scalar, and concatenates the 9 scalars.

    SB3 integration:
        Inherits from ``ActionMeanExtractor``. The concatenated actor output is
        already the Gaussian action mean, so ``CustomActorCriticPolicy`` skips
        SB3's default ``action_net``.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.obs_per_iter: int = 2

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        idx = []
        for i in range(9):
            idx.append(
                self.observation_layout.tensor(
                    (
                        self.observation_layout.slice("position", i, 1)
                        + self.observation_layout.slice("phase", i + 1, 1)
                    ),
                    self.device,
                )
            )  # + 1 on phases, as head phase is not relevant

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



class ObsLocalActJointIndependentActionHead(ActionMeanExtractor):
    """Per-action local-feedback action-mean head.

    Cfg key:
        Set ``RL.localFeedback: non-shared``. Legacy class name:
        ``localFeedbackNonShared``.

    Architecture:
        Uses the same local position/phase feature pairs as
        ``ObsLocalActJointSharedActionHead``, but creates a separate MLP for each
        action output instead of sharing one MLP across all actions.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()

        self.obs_per_iter: int = 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        idx = []
        for i in range(9):
            idx.append(
                self.observation_layout.tensor(
                    (
                        self.observation_layout.slice("position", i, 1)
                        + self.observation_layout.slice("phase", i + 1, 1)
                    ),
                    self.device,
                )
            )  # + 1 on phases, as head phase is not relevant

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


class ObsGlobalActSplitIndependentActionHead(ActionMeanExtractor):
    """Two-branch full-observation action-mean head.

    Cfg key:
        Set ``RL.localFeedback: nn3``. Legacy class name: ``nn3``.

    Architecture:
        Builds two actor MLPs over the full feature vector. The first branch
        outputs 5 action means, the second outputs 4, and both outputs are
        concatenated into the final action vector.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
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

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        x = self.policy_net_1(features)
        y = self.policy_net_2(features)
        out = torch.cat((x, y), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ObsRegionActFrontBackIndependentActionHead(ActionMeanExtractor):
    """Front/back split-observation action-mean head.

    Cfg key:
        Set ``RL.localFeedback: nn4``. Legacy class name: ``nn4``.

    Architecture:
        Splits joint positions and phases into front and back windows. One MLP
        maps the front window to 5 action means and a second MLP maps the back
        window to 4 action means.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        idx_1 = self.observation_layout.tensor(
            self.observation_layout.window(0, 5, fields=("position", "phase")),
            self.device,
        )
        idx_2 = self.observation_layout.tensor(
            self.observation_layout.window(5, 5, fields=("position", "phase")),
            self.device,
        )

        x = self.policy_net_1(torch.index_select(features, dim=1, index=idx_1))
        y = self.policy_net_2(torch.index_select(features, dim=1, index=idx_2))
        out = torch.cat((x, y), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ObsRegionActFrontBackSharedActionHead(ActionMeanExtractor):
    """Shared front/back split-observation action-mean head.

    Cfg key:
        Set ``RL.localFeedback: nn5``. Legacy class name: ``nn5``.

    Architecture:
        Uses the same 10-feature front/back windows as ``nn4``, but reuses one
        actor MLP for both windows. Each window produces 5 action means.

    SB3 integration:
        Returns final action means directly. This key is currently kept in the
        registry for documentation, but ``CustomActorCriticPolicy`` raises
        ``NotImplementedError`` for ``nn5``.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        idx_1 = self.observation_layout.tensor(
            self.observation_layout.window(0, 5, fields=("position", "phase")),
            self.device,
        )
        idx_2 = self.observation_layout.tensor(
            self.observation_layout.window(5, 5, fields=("position", "phase")),
            self.device,
        )

        x = self.policy_net_1(torch.index_select(features, dim=1, index=idx_1))
        y = self.policy_net_1(torch.index_select(features, dim=1, index=idx_2))
        out = torch.cat((x, y), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ObsWindow3ActJointPartialSharedActionHead(ActionMeanExtractor):
    """Body/tail sliding-window action-mean head.

    Cfg key:
        Set ``RL.localFeedback: nn6``. Legacy class name: ``nn6``.

    Architecture:
        Creates overlapping 6-feature windows of neighboring positions/phases.
        A shared body MLP produces the first 8 scalar actions, while a separate
        tail MLP produces the final scalar action.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        # body
        idx = []
        for i in range(8):
            idx.append(
                self.observation_layout.tensor(
                    self.observation_layout.window(
                        i, 3, fields=("position", "phase")
                    ),
                    self.device,
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


class ObsWindow3ActJointIndependentActionHead(ActionMeanExtractor):
    """Per-action sliding-window action-mean head.

    Cfg key:
        Set ``RL.localFeedback: nn7``. Legacy class name: ``nn7``.

    Architecture:
        Builds 9 separate scalar actor MLPs. Each body action sees an
        overlapping 9-feature window containing nearby positions, phases, and
        velocities; the final action reuses the tail-side window.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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
                self.observation_layout.tensor(
                    self.observation_layout.window(i, 3),
                    self.device,
                )
            )

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


class ObsCaudalLocalActJointIndependentActionHead(ActionMeanExtractor):
    """Caudal/local position-phase action-mean head.

    Cfg key:
        Set ``RL.localFeedback: caudl2``. Legacy class name: ``caudl2``.

    Architecture:
        Builds 9 scalar actor MLPs. Each action sees one joint position and the
        corresponding phase feature, giving a compact caudal/local feedback
        controller.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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
                self.observation_layout.tensor(
                    self.observation_layout.single_joint(
                        i, fields=("position", "phase")
                    ),
                    self.device,
                )
            )

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


class ObsCaudalLocalVelActJointIndependentActionHead(ActionMeanExtractor):
    """Caudal/local position-phase-velocity action-mean head.

    Cfg key:
        Set ``RL.localFeedback: caudl``. Legacy class name: ``caudl``.

    Architecture:
        Builds 9 scalar actor MLPs. Each action sees one joint position, one
        phase feature, and one velocity feature, extending ``caudl2`` with local
        velocity feedback.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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
                self.observation_layout.tensor(
                    self.observation_layout.single_joint(
                        i, fields=("position", "phase", "velocity")
                    ),
                    self.device,
                )
            )

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


class ObsWindow3VelActJointPartialSharedActionHead(ActionMeanExtractor):
    """Shared body/tail extended-window action-mean head.

    Cfg key:
        Set ``RL.localFeedback: enn7``. Legacy class name: ``enn7``.

    Architecture:
        Uses extended 9-feature windows of positions, phases, and velocities.
        One shared MLP handles body actions and a second MLP handles the tail
        action.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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
                self.observation_layout.tensor(
                    self.observation_layout.window(i, 3),
                    self.device,
                )
            )

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


class ObsGlobalActThreeRegionIndependentActionHead(ActionMeanExtractor):
    """Three-group full-observation action-mean head.

    Cfg key:
        Set ``RL.localFeedback: nn8``. Legacy class name: ``nn8``.

    Architecture:
        Builds three actor MLPs that each consume the full observation and
        output 3 action means. The three groups are concatenated into the final
        action vector.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
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


class ObsRegionActThreeRegionIndependentActionHead(ActionMeanExtractor):
    """Three-group overlapping-window action-mean head.

    Cfg key:
        Set ``RL.localFeedback: nn9``. Legacy class name: ``nn9``.

    Architecture:
        Builds three actor MLPs over overlapping 15-feature body windows. Each
        branch outputs 3 action means and the outputs are concatenated.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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
        self.idx_0 = self.observation_layout.tensor(
            self.observation_layout.window(0, 5),
            self.device,
        )

        self.idx_1 = self.observation_layout.tensor(
            self.observation_layout.window(3, 5),
            self.device,
        )

        self.idx_2 = self.observation_layout.tensor(
            self.observation_layout.window(5, 5),
            self.device,
        )

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


class ObsDriveFbActDriveFbSplitPartitionedActionHead(ActionMeanExtractor):
    """Drive/feedback split action-mean head.

    Cfg key:
        Set ``RL.localFeedback: dnn1``. Legacy class name: ``dnn1``.

    Architecture:
        Splits the actor into two branches. ``policy_net_drive`` consumes the
        first four velocity/drive-related features and outputs 2 drive actions.
        ``policy_net_fb`` consumes the remaining features and outputs the
        feedback/stretch actions.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.
        Curriculum callbacks may inspect the ``policy_net_drive`` branch.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        # body
        idx_0 = self.observation_layout.tensor(
            self.observation_layout.field("body_velocity"),
            self.device,
        )
        idx_1 = self.observation_layout.tensor(
            self.observation_layout.without(idx_0.detach().cpu().tolist()),
            self.device,
        )

        # pay attention on order or actions! It must go head to tail.
        x0 = self.policy_net_drive(torch.index_select(features, dim=1, index=idx_0))
        x1 = self.policy_net_fb(torch.index_select(features, dim=1, index=idx_1))

        out = torch.cat((x0, x1), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ObsDriveFbCurriculumActDriveFbSplitPartitionedActionHead(ActionMeanExtractor):
    """Curriculum-aware drive/feedback split action-mean head.

    Cfg key:
        Set ``RL.localFeedback: dnn2``. Legacy class name: ``dnn2``.

    Architecture:
        Uses the same drive/feedback split as ``dnn1``. During curriculum stage
        0, the drive action slice is replaced with zeros while the feedback
        branch remains active. Later stages use the learned drive branch.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.
        ``CurriculumStageCallback`` also freezes/unfreezes ``policy_net_drive``.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features: 0-9: joint positions; 10-19: phases

        drive_idx = self.observation_layout.field("body_velocity")
        idx_1 = self.observation_layout.tensor(
            self.observation_layout.without(drive_idx),
            self.device,
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
                idx_0 = self.observation_layout.tensor(drive_idx, self.device)
                x0 = self.policy_net_drive(
                    torch.index_select(features, dim=1, index=idx_0)
                )

        out = torch.cat((x0, x1), dim=1)

        assert out.shape[1] == self.action_dim  # test action dim

        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ObsDriveFbHistoryActDriveFbSplitPartitionedActionHead(ActionMeanExtractor):
    """State-history curriculum drive/feedback action-mean head.

    Cfg key:
        Set ``RL.localFeedback: dnn3``. Legacy class name: ``dnn3``.

    Architecture:
        Compresses each state-history feature with learned temporal filters,
        then applies the same curriculum-aware drive/feedback split as ``dnn2``.
        Two temporal filters are learned per base feature.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.
        The critic receives the same history-compressed features as the actor.

    Args:
        feature_dim: Flattened observation dimension including state history.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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

        drive_idx = self.observation_layout.field("body_velocity")
        self.idx_0 = self.observation_layout.tensor(
            self.observation_layout.history_indices(drive_idx, self.num_filters),
            self.device,
        )

        self.idx_1 = self.observation_layout.tensor(
            self.observation_layout.without_from_size(
                self.feature_dim * self.num_filters,
                self.idx_0.detach().cpu().tolist(),
            ),
            self.device,
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


class ObsRegionActThreeRegionPartialSharedActionHead(ActionMeanExtractor):
    """Two-shared-head three-group window action-mean head.

    Cfg key:
        Set ``RL.localFeedback: enn8``. Legacy class name: ``enn8``.

    Architecture:
        Uses three overlapping 15-feature windows. The first two windows share
        one 3-action MLP, and the tail-side window uses a second 3-action MLP.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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
        self.idx_0 = self.observation_layout.tensor(
            self.observation_layout.window(0, 5),
            self.device,
        )

        self.idx_1 = self.observation_layout.tensor(
            self.observation_layout.window(3, 5),
            self.device,
        )

        self.idx_2 = self.observation_layout.tensor(
            self.observation_layout.window(5, 5),
            self.device,
        )

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


class ObsGlobalActJointIndependentActionHead(ActionMeanExtractor):
    """Per-action full-observation action-mean head.

    Cfg key:
        Set ``RL.localFeedback: enn1``. Legacy class name: ``enn1``.

    Architecture:
        Builds 9 separate scalar actor MLPs. Each MLP sees the full observation,
        so actions are independent only at the final-head level, not by input
        locality.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
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


class ObsGlobalExtendedActThreeRegionIndependentActionHead(ActionMeanExtractor):
    """Three-group full-observation action-mean head.

    Cfg key:
        Set ``RL.localFeedback: enn2``. Legacy class name: ``enn2``.

    Architecture:
        Builds three full-observation actor MLPs. Each outputs a 3-action group,
        reducing the number of actor heads compared with ``enn1``.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
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


class ObsWindow7ActJointIndependentActionHead(ActionMeanExtractor):
    """Per-action extended-window action-mean head.

    Cfg key:
        Set ``RL.localFeedback: enn3``. Legacy class name: ``enn3``.

    Architecture:
        Builds 9 scalar actor MLPs. Each action sees an extended overlapping
        local window of positions, phases, and velocities, rather than the full
        observation.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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

        self.idx_0 = self.observation_layout.tensor(
            self.observation_layout.window(0, 7),
            self.device,
        )
        self.idx_1 = self.observation_layout.tensor(
            self.observation_layout.window(1, 7),
            self.device,
        )
        self.idx_2 = self.observation_layout.tensor(
            self.observation_layout.window(2, 7),
            self.device,
        )
        self.idx_3 = self.observation_layout.tensor(
            self.observation_layout.window(3, 7),
            self.device,
        )

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


class ObsWindow5ActJointIndependentActionHead(ActionMeanExtractor):
    """Per-action 15-feature window action-mean head.

    Cfg key:
        Set ``RL.localFeedback: enn4``. Legacy class name: ``enn4``.

    Architecture:
        Builds 9 scalar actor MLPs over overlapping 15-feature windows. Boundary
        actions reuse the nearest available window.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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
                self.observation_layout.tensor(
                    self.observation_layout.window(i, 5),
                    self.device,
                )
            )

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


class ObsWindow5ActJointPartialSharedActionHead(ActionMeanExtractor):
    """Shared body/tail window action-mean head.

    Cfg key:
        Set ``RL.localFeedback: enn6``. Legacy class name: ``enn6``.

    Architecture:
        Uses four scalar actor MLPs. One MLP is shared across the central body
        windows, while separate MLPs handle boundary and tail-side outputs.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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
                self.observation_layout.tensor(
                    self.observation_layout.window(i, 5),
                    self.device,
                )
            )

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


class ObsWindow7ActThreeRegionIndependentActionHead(ActionMeanExtractor):
    """Three-group extended-window action-mean head.

    Cfg key:
        Set ``RL.localFeedback: enn5``. Legacy class name: ``enn5``.

    Architecture:
        Builds three actor MLPs over extended overlapping windows. Each branch
        outputs 3 action means, and the groups are concatenated.

    SB3 integration:
        Returns final action means directly. SB3's ``action_net`` is bypassed.

    Args:
        feature_dim: Feature dimension produced by SB3's feature extractor.
        action_dim: Number of continuous actions expected by the environment.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.observation_layout = ObservationLayout.from_conf()
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
        self.idx_0 = self.observation_layout.tensor(
            self.observation_layout.window(0, 7),
            self.device,
        )
        self.idx_1 = self.observation_layout.tensor(
            self.observation_layout.window(2, 7),
            self.device,
        )
        self.idx_2 = self.observation_layout.tensor(
            self.observation_layout.window(3, 7),
            self.device,
        )

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


NETWORK_REGISTRY = {
    # Config key -> descriptive class. Comments keep the previous class names.
    "shared": ObsLocalActJointSharedActionHead,  # localFeedbackShared
    "non-shared": ObsLocalActJointIndependentActionHead,  # localFeedbackNonShared
    "nn3": ObsGlobalActSplitIndependentActionHead,
    "nn4": ObsRegionActFrontBackIndependentActionHead,
    "nn5": ObsRegionActFrontBackSharedActionHead,
    "nn6": ObsWindow3ActJointPartialSharedActionHead,
    "nn7": ObsWindow3ActJointIndependentActionHead,
    "caudl": ObsCaudalLocalVelActJointIndependentActionHead,
    "caudl2": ObsCaudalLocalActJointIndependentActionHead,
    "nn8": ObsGlobalActThreeRegionIndependentActionHead,
    "nn9": ObsRegionActThreeRegionIndependentActionHead,
    "enn1": ObsGlobalActJointIndependentActionHead,
    "enn2": ObsGlobalExtendedActThreeRegionIndependentActionHead,
    "enn3": ObsWindow7ActJointIndependentActionHead,
    "enn4": ObsWindow5ActJointIndependentActionHead,
    "enn5": ObsWindow7ActThreeRegionIndependentActionHead,
    "enn6": ObsWindow5ActJointPartialSharedActionHead,
    "enn7": ObsWindow3VelActJointPartialSharedActionHead,
    "enn8": ObsRegionActThreeRegionPartialSharedActionHead,
    "dnn1": ObsDriveFbActDriveFbSplitPartitionedActionHead,
    "dnn2": ObsDriveFbCurriculumActDriveFbSplitPartitionedActionHead,
    "dnn3": ObsDriveFbHistoryActDriveFbSplitPartitionedActionHead,
}

NETWORK_ALIASES = {
    # Very local position/phase controllers.
    "obs_local__act_joint__shared": "shared",
    "obs_local__act_joint__independent": "non-shared",
    # Caudal-local controllers are per-joint in the current implementation.
    "obs_caudal_local__act_joint__independent": "caudl2",
    "obs_caudal_local_vel__act_joint__independent": "caudl",
    # Sliding local windows. The window number is the number of neighboring
    # body positions represented per position/phase/velocity group.
    "obs_window3__act_joint__partial_shared": "nn6",
    "obs_window3__act_joint__independent": "nn7",
    "obs_window3_vel__act_joint__partial_shared": "enn7",
    "obs_window5__act_joint__independent": "enn4",
    "obs_window5__act_joint__partial_shared": "enn6",
    "obs_window7__act_joint__independent": "enn3",
    # Regional action heads.
    "obs_region__act_frontback__independent": "nn4",
    "obs_region__act_frontback__shared": "nn5",
    "obs_region__act_three_region__independent": "nn9",
    "obs_region__act_three_region__partial_shared": "enn8",
    "obs_window7__act_three_region__independent": "enn5",
    # Global observation heads.
    "obs_global__act_split__independent": "nn3",
    "obs_global__act_three_region__independent": "nn8",
    "obs_global_extended__act_three_region__independent": "enn2",
    "obs_global__act_joint__independent": "enn1",
    # Drive/feedback split heads.
    "obs_drive_fb__act_drive_fb_split__partitioned": "dnn1",
    "obs_drive_fb_curriculum__act_drive_fb_split__partitioned": "dnn2",
    "obs_drive_fb_history__act_drive_fb_split__partitioned": "dnn3",
}

STATE_HISTORY_REGISTRY = {
    "sh1": PerFeatureStateHistoryExtractor,
    "sh2": GroupedStateHistoryExtractor,
}


def _valid_local_feedback_message() -> str:
    old_names = ", ".join(sorted(NETWORK_REGISTRY))
    readable_aliases = ", ".join(sorted(NETWORK_ALIASES))
    return f"valid old names: {old_names}; valid readable aliases: {readable_aliases}"

# Legacy aliases for reading old experiment configs, checkpoints, and notes.
CustomNetwork = StandardConfigurableExtractor
sh1 = PerFeatureStateHistoryExtractor
sh2 = GroupedStateHistoryExtractor
localFeedbackShared = ObsLocalActJointSharedActionHead
localFeedbackNonShared = ObsLocalActJointIndependentActionHead
nn3 = ObsGlobalActSplitIndependentActionHead
nn4 = ObsRegionActFrontBackIndependentActionHead
nn5 = ObsRegionActFrontBackSharedActionHead
nn6 = ObsWindow3ActJointPartialSharedActionHead
nn7 = ObsWindow3ActJointIndependentActionHead
caudl = ObsCaudalLocalVelActJointIndependentActionHead
caudl2 = ObsCaudalLocalActJointIndependentActionHead
enn1 = ObsGlobalActJointIndependentActionHead
enn2 = ObsGlobalExtendedActThreeRegionIndependentActionHead
enn3 = ObsWindow7ActJointIndependentActionHead
enn4 = ObsWindow5ActJointIndependentActionHead
enn5 = ObsWindow7ActThreeRegionIndependentActionHead
enn6 = ObsWindow5ActJointPartialSharedActionHead
enn7 = ObsWindow3VelActJointPartialSharedActionHead
enn8 = ObsRegionActThreeRegionPartialSharedActionHead
nn8 = ObsGlobalActThreeRegionIndependentActionHead
nn9 = ObsRegionActThreeRegionIndependentActionHead
dnn1 = ObsDriveFbActDriveFbSplitPartitionedActionHead
dnn2 = ObsDriveFbCurriculumActDriveFbSplitPartitionedActionHead
dnn3 = ObsDriveFbHistoryActDriveFbSplitPartitionedActionHead

# Previous descriptive class names kept as aliases for compatibility.
SharedLocalFeedbackActionHead = ObsLocalActJointSharedActionHead
PerJointLocalFeedbackActionHead = ObsLocalActJointIndependentActionHead
TwoBranchFullObservationActionHead = ObsGlobalActSplitIndependentActionHead
FrontBackSplitObservationActionHead = ObsRegionActFrontBackIndependentActionHead
SharedFrontBackSplitActionHead = ObsRegionActFrontBackSharedActionHead
BodyTailSlidingWindowActionHead = ObsWindow3ActJointPartialSharedActionHead
PerJointSlidingWindowActionHead = ObsWindow3ActJointIndependentActionHead
CaudalPhasePositionActionHead = ObsCaudalLocalActJointIndependentActionHead
CaudalStateVelocityActionHead = ObsCaudalLocalVelActJointIndependentActionHead
BodyTailSharedExtendedWindowActionHead = ObsWindow3VelActJointPartialSharedActionHead
ThreeGroupFullObservationActionHead = ObsGlobalActThreeRegionIndependentActionHead
ThreeGroupOverlappingWindowActionHead = ObsRegionActThreeRegionIndependentActionHead
DriveFeedbackActionHead = ObsDriveFbActDriveFbSplitPartitionedActionHead
CurriculumDriveFeedbackActionHead = (
    ObsDriveFbCurriculumActDriveFbSplitPartitionedActionHead
)
HistoryCurriculumDriveFeedbackActionHead = (
    ObsDriveFbHistoryActDriveFbSplitPartitionedActionHead
)
TwoSharedThreeGroupWindowActionHead = ObsRegionActThreeRegionPartialSharedActionHead
PerActionFullObservationActionHead = ObsGlobalActJointIndependentActionHead
ThreeGroupFullObservationSharedActionHead = (
    ObsGlobalExtendedActThreeRegionIndependentActionHead
)
PerActionExtendedWindowActionHead = ObsWindow7ActJointIndependentActionHead
PerActionFifteenFeatureWindowActionHead = ObsWindow5ActJointIndependentActionHead
SharedBodyTailWindowActionHead = ObsWindow5ActJointPartialSharedActionHead
ThreeGroupExtendedWindowActionHead = ObsWindow7ActThreeRegionIndependentActionHead


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
