import torch
import torch as th
from typing import Dict, Iterable, List, Sequence, Tuple

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
