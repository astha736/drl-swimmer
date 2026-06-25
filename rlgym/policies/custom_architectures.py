import numpy as np
import torch
import torch as th
import torch.nn as nn

import conf
from .common import ActionMeanExtractor
from .observation_layout import ObservationLayout


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
