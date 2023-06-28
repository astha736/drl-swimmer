"""Simulation"""

from mimetypes import init
import os
from textwrap import wrap
import warnings
import traceback
from enum import Enum
from typing import List
import numpy as np
import random

from utils import simulation

from dm_control.rl.control import Environment, PhysicsError
from dm_env import TimeStep, StepType

from farms_core import pylog
from farms_mujoco.simulation.application import FarmsApplication
from farms_mujoco.simulation.task import TaskCallback
from farms_mujoco.simulation.mjcf import euler2mjcquat
from farms_sim.simulation import postprocessing_from_clargs


import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

import csv

from utils.limbless_spawn import RobotInitialState
from utils.limbless_oscillator import RobotInitialOscillator

from utils import utils
import conf


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ActionType(Enum):
    STRETCH = 1  # stretch
    CONTACT = 2  # contact
    DRIVE = 3  # drive
    STRETCH_BIAS = 4 # bias with sin, cos


class ObservationType(Enum):
    JOINT_POSITION = 1
    REACTION_X = 2
    REACTION_Y = 3
    REACTION_Z = 4
    REACTION_XY = 5
    REACTION_XYZ = 6
    PHASES = 7
    VELOCITIES = 8
    AMPLITUDES = 9


class ActionChoice:
    """_summary_

    Keeping the action lower and upper values between -1 and 1
    """

    action_output_scale = {
        ActionType.STRETCH: 30,
        ActionType.CONTACT: 10,
        ActionType.DRIVE: [2.5, 4.5],
        ActionType.STRETCH_BIAS: 30,
    }

    def __init__(self, action_list: List[ActionType], n_body_joints: int = 10):
        self.action_list = action_list
        self.n_body_joints = n_body_joints
        self.action_length = {}  # 'dict{ActionType, int}'

        # create action spaces
        self.action_space, self.n_act = self.create_action_space()

    def action_bound_STRETCH(self):
        """Stretch can be caudal connected or rostral connected

        Thus one of the joint in the body can not be used as sensor
        for a given connectivity direction
        """
        if not conf.CONF["robot_arch"]["s_local_weight"] == None:
            self.action_length[ActionType.STRETCH] = self.n_body_joints
        else:
            self.action_length[ActionType.STRETCH] = self.n_body_joints - 1
        low = np.array([-1] * self.action_length[ActionType.STRETCH])
        high = np.array([1] * self.action_length[ActionType.STRETCH])
        return low, high
    
    def action_bound_STRETCH_BIAS(self):
        if not conf.CONF["robot_arch"]["s_local_weight"] == None:
            self.action_length[ActionType.STRETCH_BIAS] = self.n_body_joints * 2
        else:
            self.action_length[ActionType.STRETCH_BIAS] = (self.n_body_joints - 1) * 2
        low = np.array([-1] * self.action_length[ActionType.STRETCH_BIAS])
        high = np.array([1] * self.action_length[ActionType.STRETCH_BIAS])
        return low, high

    def action_bound_CONTACT(self):
        """Contacts are usually same as joint"""

        self.action_length[ActionType.CONTACT] = self.n_body_joints
        low = np.array([-1] * self.action_length[ActionType.CONTACT])
        high = np.array([1] * self.action_length[ActionType.CONTACT])

        return low, high

    def action_bound_DRIVE(self):
        self.action_length[ActionType.DRIVE] = 2  # 2 drives
        low = np.array([-1] * self.action_length[ActionType.DRIVE])
        high = np.array([1] * self.action_length[ActionType.DRIVE])

        return low, high

    def get_action_bound(self, action: ActionType):
        switcher = {
            ActionType.STRETCH: self.action_bound_STRETCH,
            ActionType.CONTACT: self.action_bound_CONTACT,
            ActionType.DRIVE: self.action_bound_DRIVE,
            ActionType.STRETCH_BIAS: self.action_bound_STRETCH_BIAS,
        }

        return switcher.get(action, "Invalid Action Type")

    def create_action_space(self):
        low_bound, high_bound = [], []
        for action in self.action_list:
            low, high = self.get_action_bound(action)()
            low_bound += [low]
            high_bound += [high]

        low_bound = np.concatenate((low_bound), axis=0)
        high_bound = np.concatenate((high_bound), axis=0)

        if np.shape(low_bound)[0] != np.shape(high_bound)[0]:
            raise ValueError(
                "[ActionChoice] lower and upper bound shape are not the same"
            )

        return spaces.Box(low=low_bound, high=high_bound), np.shape(low_bound)[0]

    def set_action_STRETCH(self, action, network_parameters, iteration, data_states):
        action = action * ActionChoice.action_output_scale[ActionType.STRETCH]

        robot_parameters = network_parameters.joints2osc_map.weights.array

        for i, action_val in enumerate(action):
            robot_parameters[i * 2 + 0] = action_val  # left oscillator assignment
            robot_parameters[i * 2 + 1] = action_val * -1  # right oscillator assignment
        pass

    def set_action_STRETCH_BIAS(self, action, network_parameters, iteration, data_states):
        action = action * ActionChoice.action_output_scale[ActionType.STRETCH_BIAS]

        robot_parameters = network_parameters.joints2osc_map.weights.array

        phases_left = np.array(data_states.phases(iteration))[
                        conf.LEFT_OSCILLATOR_INDEXES
                    ]
        
        phases_right = np.array(data_states.phases(iteration))[
                        conf.RIGHT_OSCILLATOR_INDEXES
                    ]
        
        # two actions for each joint
        for i, j in enumerate(range(0, len(action), 2)):
            action_biased_left = action[j] * np.cos(phases_left[i]) + action[j+1] * np.sin(phases_left[i])
            action_biased_right = - action[j] * np.cos(phases_right[i]) - action[j+1] * np.sin(phases_right[i])
            robot_parameters[i * 2 + 0] = action_biased_left  # left oscillator assignment
            robot_parameters[i * 2 + 1] = action_biased_right  # right oscillator assignment
        pass

    def set_action_CONTACT(self, action, network_parameters, iteration, data_states):
        action = action * ActionChoice.action_output_scale[ActionType.STRETCH]

        # ASTHA BUG FIX
        robot_parameters = network_parameters.contact2osc_map.weights.array

        for i, action_val in enumerate(action):
            robot_parameters[i * 2 + 0] = action_val  # left oscillator assignment
            robot_parameters[i * 2 + 1] = action_val * -1  # right oscillator assignment

        pass

    def set_action_DRIVE(self, action, network_parameters, iteration, data_states):
        # network_parameters = self.sim.task.data.network
        # setting data.network.drives.array

        # rescale action
        action = ((action - (-1)) / 2) * (
            ActionChoice.action_output_scale[ActionType.DRIVE][1]
            - ActionChoice.action_output_scale[ActionType.DRIVE][0]
        ) + ActionChoice.action_output_scale[ActionType.DRIVE][0]

        # set action
        network_parameters.drives.array[iteration][0] = action[0]
        network_parameters.drives.array[iteration][1] = action[1]

        pass

    def set_action_switch(self, observation: ActionType):
        switcher = {
            ActionType.STRETCH: self.set_action_STRETCH,
            ActionType.CONTACT: self.set_action_CONTACT,
            ActionType.DRIVE: self.set_action_DRIVE,
            ActionType.STRETCH_BIAS: self.set_action_STRETCH_BIAS,      
        }

        return switcher.get(observation, "Invalid observation Type")

    def set_action(self, actions, network_parameters, iteration: int, data_states):
        index = 0
        for action_type in self.action_list:
            action_len = self.action_length[action_type]
            action_slice = actions[index : index + action_len]
            self.set_action_switch(action_type)(
                action_slice, network_parameters, iteration, data_states
            )
            index += action_len
        return


class ObservationChoice:
    """_summary_

    Keeping the observation lower and upper values between -1 and 1

    TODO: save n_obs for each observation type
    """

    def __init__(
        self, observation_list: List[ObservationType], n_body_joints: int = 10
    ):
        self.observation_list = observation_list
        self.n_body_joints = n_body_joints
        self.observation_space, self.n_obs = self.create_observation_space()

    def observation_bound_JOINT_POSITION(self):
        """JOINT POSITION"""
        low = np.array([-np.inf] * (self.n_body_joints))
        high = np.array([np.inf] * (self.n_body_joints))
        return low, high

    def observation_bound_REACTION_X(self):
        """REACTION X direction"""
        low = np.array([-np.inf] * (self.n_body_joints + 1))
        high = np.array([np.inf] * (self.n_body_joints + 1))
        return low, high

    def observation_bound_REACTION_Y(self):
        """REACTION Y direction"""
        low = np.array([-np.inf] * (self.n_body_joints + 1))
        high = np.array([np.inf] * (self.n_body_joints + 1))
        return low, high

    def observation_bound_REACTION_Z(self):
        """REACTION Z direction"""
        low = np.array([-np.inf] * (self.n_body_joints + 1))
        high = np.array([np.inf] * (self.n_body_joints + 1))
        return low, high

    def observation_bound_REACTION_XY(self):
        """REACTION Y direction"""
        low = np.array([-np.inf] * (self.n_body_joints + 1))
        high = np.array([np.inf] * (self.n_body_joints + 1))
        return low, high

    def observation_bound_REACTION_XYZ(self):
        """REACTION Z direction"""
        low = np.array([-np.inf] * (self.n_body_joints + 1))
        high = np.array([np.inf] * (self.n_body_joints + 1))
        return low, high

    def observation_bound_PHASES(self):
        """PHASES"""
        low = np.array([-np.inf] * (self.n_body_joints))
        high = np.array([np.inf] * (self.n_body_joints))
        return low, high

    def observation_bound_AMPLITUDES(self):
        """AMPLITUDES"""
        low = np.array([-np.inf] * (self.n_body_joints))
        high = np.array([np.inf] * (self.n_body_joints))
        return low, high

    def observation_bound_VELOCITIES(self):
        """VELOCITIES"""
        low = np.array([-np.inf] * (4))  # 2 * target + 2 * current
        high = np.array([np.inf] * (4))  # 2 * target + 2 * current
        return low, high

    def get_observation_bound(self, observation: ObservationType):
        switcher = {
            ObservationType.JOINT_POSITION: self.observation_bound_JOINT_POSITION,
            ObservationType.REACTION_X: self.observation_bound_REACTION_X,
            ObservationType.REACTION_Y: self.observation_bound_REACTION_Y,
            ObservationType.REACTION_Z: self.observation_bound_REACTION_Z,
            ObservationType.REACTION_XY: self.observation_bound_REACTION_XY,
            ObservationType.REACTION_XYZ: self.observation_bound_REACTION_XYZ,
            ObservationType.PHASES: self.observation_bound_PHASES,
            ObservationType.AMPLITUDES: self.observation_bound_AMPLITUDES,
            ObservationType.VELOCITIES: self.observation_bound_VELOCITIES,
        }

        return switcher.get(observation, "Invalid observation Type")

    def create_observation_space(self):
        low_bound, high_bound = [], []
        for observation in self.observation_list:
            low, high = self.get_observation_bound(observation)()
            low_bound += [low]
            high_bound += [high]

        low_bound = np.concatenate((low_bound), axis=0)
        high_bound = np.concatenate((high_bound), axis=0)

        if np.shape(low_bound)[0] != np.shape(high_bound)[0]:
            raise ValueError(
                "[ObservationChoice] lower and upper bound shape are not the same"
            )

        return spaces.Box(low=low_bound, high=high_bound), np.shape(low_bound)[0]

    def extract_observation_JOINT_POSITION(self, data_sensors, data_states, iteration):
        joints_pos = np.array(data_sensors.joints.positions(iteration=iteration))

        return joints_pos

    def extract_observation_VELOCITIES(self, data_sensors, data_states, iteration):
        com_velocity = np.array(data_sensors.links.global_com_velocity(iteration))[0:2]

        target_velocity = np.array(
            conf.CONF["RL"]["target_velocity"],
        )

        return np.concatenate((com_velocity, target_velocity))

    def extract_observation_AMPLITUDES(self, data_sensors, data_states, iteration):
        amplitudes_right = np.array(data_states.amplitudes(iteration))[
            conf.RIGHT_OSCILLATOR_INDEXES
        ]
        return amplitudes_right

    def extract_observation_PHASES(self, data_sensors, data_states, iteration):
        # only return right oscillators for now.
        # this is technical not correct, as left and right oscillators are not initialized ideally
        # so they need some time to sync
        # however, I want to reduce input observation space for now
        match conf.CONF["RL"]["phase_preprocessing"]:
            case "sin":
                phases_right = np.sin(
                    np.array(data_states.phases(iteration))[
                        conf.RIGHT_OSCILLATOR_INDEXES
                    ]
                )
            case "cos":
                phases_right = np.cos(
                    np.array(data_states.phases(iteration))[
                        conf.RIGHT_OSCILLATOR_INDEXES
                    ]
                )
            case "mod":
                phases_right = np.mod(
                    np.array(data_states.phases(iteration))[
                        conf.RIGHT_OSCILLATOR_INDEXES
                    ],
                    2 * np.pi,
                )
            case _:
                raise ValueError("Unknown phase preprocessing method")
        return phases_right

    def extract_observation_REACTION_X(self, data_sensors, data_states, iteration):
        data_reaction_x = np.array(data_sensors.contacts.array[iteration, :, 0])
        isNaN = np.isnan(data_reaction_x).any()
        if isNaN:
            warnings.warn(
                bcolors.WARNING
                + "NaN values in contact forces at itr {}".format(iteration)
                + bcolors.ENDC
            )
        np.nan_to_num(data_reaction_x, copy=False, nan=0.0, posinf=0.0, neginf=-0.0)
        return data_reaction_x

    def extract_observation_REACTION_Y(self, data_sensors, data_states, iteration):
        data_reaction_y = np.array(data_sensors.contacts.array[iteration, :, 1])
        isNaN = np.isnan(data_reaction_y).any()
        if isNaN:
            warnings.warn(
                bcolors.WARNING
                + "NaN values in contact forces at itr {}".format(iteration)
                + bcolors.ENDC
            )
        np.nan_to_num(data_reaction_y, copy=False, nan=0.0, posinf=0.0, neginf=-0.0)
        return data_reaction_y

    def extract_observation_REACTION_Z(self, data_sensors, data_states, iteration):
        data_reaction_z = np.array(data_sensors.contacts.array[iteration, :, 2])
        isNaN = np.isnan(data_reaction_z).any()
        if isNaN:
            warnings.warn(
                bcolors.WARNING
                + "NaN values in contact forces at itr {}".format(iteration)
                + bcolors.ENDC
            )
        np.nan_to_num(data_reaction_z, copy=False, nan=0.0, posinf=0.0, neginf=-0.0)
        return data_reaction_z

    def extract_observation_REACTION_XY_NORM(
        self, data_sensors, data_states, iteration
    ):
        data_reaction_xy = np.array(data_sensors.contacts.array[iteration, :, 0:2])
        isNaN = np.isnan(data_reaction_xy).any()
        if isNaN:
            warnings.warn(
                bcolors.WARNING
                + "NaN values in contact forces at itr {}".format(iteration)
                + bcolors.ENDC
            )
        np.nan_to_num(data_reaction_xy, copy=False, nan=0.0, posinf=0.0, neginf=-0.0)
        data_reaction_xy_norm = np.linalg.norm(data_reaction_xy, axis=1)
        return data_reaction_xy_norm

    def extract_observation_REACTION_XYZ_NORM(
        self, data_sensors, data_states, iteration
    ):
        data_reaction_xyz = np.array(data_sensors.contacts.array[iteration, :, 0:3])
        isNaN = np.isnan(data_reaction_xyz).any()
        if isNaN:
            warnings.warn(
                bcolors.WARNING
                + "NaN values in contact forces at itr {}".format(iteration)
                + bcolors.ENDC
            )
        np.nan_to_num(data_reaction_xyz, copy=False, nan=0.0, posinf=0.0, neginf=-0.0)
        data_reaction_xyz_norm = np.linalg.norm(data_reaction_xyz, axis=1)
        return data_reaction_xyz_norm

    def extract_observation(self, observation: ObservationType):
        switcher = {
            ObservationType.JOINT_POSITION: self.extract_observation_JOINT_POSITION,
            ObservationType.REACTION_X: self.extract_observation_REACTION_X,
            ObservationType.REACTION_Y: self.extract_observation_REACTION_Y,
            ObservationType.REACTION_Z: self.extract_observation_REACTION_Z,
            ObservationType.REACTION_XY: self.extract_observation_REACTION_XY_NORM,
            ObservationType.REACTION_XYZ: self.extract_observation_REACTION_XYZ_NORM,
            ObservationType.PHASES: self.extract_observation_PHASES,
            ObservationType.AMPLITUDES: self.extract_observation_AMPLITUDES,
            ObservationType.VELOCITIES: self.extract_observation_VELOCITIES,
        }

        return switcher.get(observation, "Invalid observation Type")

    def get_observation(self, data_sensors, data_states, iteration: int):
        observations_list = []
        for observation in self.observation_list:
            observation_val = self.extract_observation(observation)(
                data_sensors, data_states, iteration
            )
            observations_list += [observation_val]

        observations_numpy = np.concatenate((observations_list), axis=0)

        return observations_numpy


class FarmsGym(gym.Env):
    """Farms Gym environment.

    @brief: This class is the main class for the gym environment. It is used to create the environment and to step through it.

    @description: Used for training only (no testing); it is a wrapper around everything in farms; implements a custom step-function
    https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html

    @todo: Add reward as a choice as well

    """

    def __init__(
        self,
        timestep,
        observation_choice: ObservationChoice,
        action_choice: ActionChoice,
        animat_options,
        arena_options,
        sim_options,
        simulator,
        is_test_env: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.observation_choice = observation_choice
        self.action_choice = action_choice
        self.observation_space = observation_choice.observation_space
        self.n_obs = observation_choice.n_obs

        self.action_space = action_choice.action_space

        self.n_act = action_choice.n_act
        self.timestep = timestep

        self.animat_options = animat_options
        self.arena_options = arena_options
        self.sim_options = sim_options
        self.simulator = simulator

        self.reward = None
        self.info = None
        self.done = None
        self.observation = None
        self.random_times = 0

        self.jointPosLastEpisode = None

        self.notion = kwargs.pop("notion", None)

        self.is_test_env = is_test_env

        if self.is_test_env:
            self.log_fb_weights = []

        self.sim, _ = simulation.setup_simulation(
            self.animat_options,
            self.arena_options,
            self.sim_options,
            self.simulator,
            callbacks=[],
        )

    def get_observations(
        data_sensors, data_states, iteration: int, observation_choice: ObservationChoice
    ):
        """get observation

        AgnathaX: Observation space is given by the observation_choice (ObservationChoice) which contains a list of
        observations for the given experiment

        """
        return observation_choice.get_observation(
            data_sensors=data_sensors, data_states=data_states, iteration=iteration
        )

    def compute_reward(data_sensors, iteration):
        """used for training and testing"""

        # forward way x
        curr_x = np.array(data_sensors.links.global_com_position(iteration))[0]
        prev_x = (
            curr_x
            if (iteration == 0)
            else np.array(data_sensors.links.global_com_position(iteration - 1)[0])
        )
        forward_x = curr_x - prev_x  # range of 0.003 per step

        # torques
        cmd_torques = np.sum(
            np.abs(np.array(data_sensors.joints.cmd_torques())[iteration])
        )  # range of ~3-4 per step for 001
        # Astha said active_torques is good for bioinspiration
        active_torques = np.array(data_sensors.joints.active_torques())[
            iteration
        ]  # range of ~1-2 per step for 001
        active_torques_prev = (
            np.array(data_sensors.joints.active_torques())[iteration - 1]
            if iteration > 0
            else 0.0
        )
        active_torque = np.sum(np.abs(active_torques))  # range of ~7 per step for 001
        active_torque_diff = np.sum(
            np.abs(active_torques - active_torques_prev)
        )  # range of ~0.3 per step for 001

        # power
        joints_power = np.sum(
            data_sensors.joints.sum_power_joints_timestep()[iteration]
        )

        # forward way COM; positive if forward_x is positive
        curr_com = np.array(data_sensors.links.global_com_position(iteration))
        prev_com = (
            curr_com
            if (iteration == 0)
            else np.array(data_sensors.links.global_com_position(iteration - 1))
        )
        forward_com = np.sign(forward_x) * np.linalg.norm(
            curr_com - prev_com
        )  # range of 0.003 per step for 001

        # speed
        speed_com = np.linalg.norm(
            np.array(data_sensors.links.global_com_velocity(iteration))
        )

        # velocity
        velocity_com = np.array(data_sensors.links.global_com_velocity(iteration))[0:2]

        # sign forward
        head_pos = np.array(data_sensors.links.com_position(iteration=iteration,link_i = 0))[0:2]
        tail_pos = np.array(data_sensors.links.com_position(iteration=iteration,link_i = 10))[0:2]
        tail_head_vec = head_pos - tail_pos
        sign_fwd = np.sign(np.dot(velocity_com, tail_head_vec) + 0.1) # ~100°

        reward = 0.0
        if "vel_com" in conf.CONF["RL"]["RewardFnc"]:
            reward += (
                conf.CONF["RL"]["RewardFnc"]["vel_com"] * speed_com * sign_fwd
            )
        if "joints_power" in conf.CONF["RL"]["RewardFnc"]:
            reward += conf.CONF["RL"]["RewardFnc"]["joints_power"] * joints_power
        if "forward_x" in conf.CONF["RL"]["RewardFnc"]:
            reward += conf.CONF["RL"]["RewardFnc"]["forward_x"] * forward_x
        if "cmd_torques" in conf.CONF["RL"]["RewardFnc"]:
            reward += conf.CONF["RL"]["RewardFnc"]["cmd_torques"] * cmd_torques
        if "active_torques" in conf.CONF["RL"]["RewardFnc"]:
            reward += conf.CONF["RL"]["RewardFnc"]["active_torques"] * active_torque
        if "healthy" in conf.CONF["RL"]["RewardFnc"]:
            reward += conf.CONF["RL"]["RewardFnc"]["healthy"]
        # TODO catch error if target_speed || speed_error is not defined
        if (
            "speed_error" in conf.CONF["RL"]["RewardFnc"]
            and "target_speed" in conf.CONF["RL"]
        ):
            reward += conf.CONF["RL"]["RewardFnc"]["speed_error"] * (
                np.abs(speed_com - conf.CONF["RL"]["target_speed"])
            )
        if "active_torque_diff" in conf.CONF["RL"]["RewardFnc"]:
            reward += (
                conf.CONF["RL"]["RewardFnc"]["active_torque_diff"] * active_torque_diff
            )
        if "forward_com" in conf.CONF["RL"]["RewardFnc"]:
            reward += conf.CONF["RL"]["RewardFnc"]["forward_com"] * forward_com
        if "velocity_error" in conf.CONF["RL"]["RewardFnc"]:
            reward += conf.CONF["RL"]["RewardFnc"]["velocity_error"] * np.sum(
                np.abs(velocity_com - conf.CONF["RL"]["target_velocity"])
            )

        return reward

    def set_action(
        action, network_parameters, action_choice: ActionChoice, iteration: int, data_states
    ):
        """Apply the computed action to the concerned variables"""
        if np.isnan(action).any():
            raise ValueError("Action is nan")

        if (action > 1).any() or (action < -1).any():
            raise ValueError("Action is out of bounce")

        if action is None:
            raise ValueError("should not be allowed")

        action_choice.set_action(action, network_parameters, iteration, data_states)
        return

    def step(self, action):
        """Performs a step on the environment"""

        # this MUST be defined here!
        iteration = self.sim.task.iteration
        if action is None:
            print("should not be allowed")

        FarmsGym.set_action(
            action=action,
            network_parameters=self.sim.task.data.network,
            action_choice=self.action_choice,
            iteration=iteration,
            data_states = self.sim.task.data.state,
        )

        # @ASTHA makes simulation go forward # @CHECK
        env_step = self.sim._env.step(
            action=None
        )  # Take control of the env; used instead of sim.run

        self.observation = FarmsGym.get_observations(
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=iteration,
            observation_choice=self.observation_choice,
        )

        # REWARD
        self.reward = FarmsGym.compute_reward(
            self.sim.task.data.sensors,
            iteration,
        )

        self.done = False
        if env_step.step_type == StepType.LAST:
            self.done = True  # end of episode

        curr_x = np.array(
            self.sim.task.data.sensors.links.global_com_position(iteration)
        )[0]
        start_x = np.array(self.sim.task.data.sensors.links.global_com_position(0))[0]
        if curr_x < start_x - 0.2 and conf.CONF["RL"]["useEarlyTerm"] == True:
            self.done = True  # early termination on backwards movement

        if self.done:
            self.jointPosLastEpisode = np.copy(np.array(self.sim.task.data.sensors.joints.positions(iteration=iteration)))
        
        if self.is_test_env:
            self.log_fb_weights.append(
                np.array(self.sim.task.data.network.joints2osc_map.weights.array)
            )

        if self.done and self.is_test_env:
            utils.save_performance_metrics(
                self.sim,
                self.timestep,
                self.sim_options.n_iterations,
            )
            fb_weights = np.array(self.log_fb_weights)
            _times = np.arange(
                0,
                self.timestep * self.sim_options.n_iterations,
                self.timestep,
            )
            fig = plt.figure(f"Feedback weights")
            for i in [0, 2, 4, 6, 8]:
                plt.plot(
                    _times,
                    fb_weights[:, i],
                    label=f"Weight joint {i}",
                )
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                borderaxespad=0,
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Feedback weight value")
            plt.grid(True)

            with PdfPages(
                os.path.join(conf.LOG_DIR_RESULTS, "performance_plots_test_env.pdf")
            ) as pdf:
                pdf.savefig(fig, bbox_inches="tight")
            if self.sim_options.record == True:
                postprocessing_from_clargs(
                    sim=self.sim,
                    video_name=os.path.join(conf.LOG_DIR_RESULTS, "best_model.mp4"),
                )

        return self.observation, self.reward, self.done, self.info

    def reset(self):
        """reset episode procedure

        Description:  This is used as an opportunity to set the new changes for episode.
        This will help in randomizing episode for robust training

        Note: There are two kind of changes, one that happens to the sdf's (mjcf_model) [mujoco]
        and other that happens to the state of the system [farms]. For the state of the system,
        the animat_option needs to be changed and then a reset() needs to called on sim._env

        sim._env.reset() calls reset on the task and environment wrapper. check initialize_episode()
        (most likely in ExperimentTask or Environment)
        as the function is called through sim._env.reset() calls

        """

        # self.sim, _ = simulation.setup_simulation(
        #     self.animat_options,
        #     self.arena_options,
        #     self.sim_options,
        #     self.simulator,
        #     callbacks = []
        # )

        # !!! oscillator states are reset manually in agnathax_control/network.py !!!

        if conf.CONF["RL"]["useRandStartCond"] =="jointPosEndLastEpisode" and not self.jointPosLastEpisode is None:
            RobotInitialState.set_user_defined_shape_pose(
                animat_options=self.sim.task.animat_options,
                shape_pose = self.jointPosLastEpisode
            )
            
            # self.sim.task.data.sensors.joints.positions(0) = self.jointPosLastEpisode
        elif conf.CONF["RL"]["useRandStartCond"]:
            pass
            # RobotInitialState.set_random_shape_pose(
            #     animat_options=self.sim.task.animat_options
            # )
        else:
            pass

        self.sim._env.reset()

        self.observation = FarmsGym.get_observations(
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=0,
            observation_choice=self.observation_choice,
        )

        # for internal use?
        self.info = {}
        self.done = False
        self.reward = 0

        if self.is_test_env:
            self.log_fb_weights = []

        return self.observation

    def render(self, mode="rgb_array", height=480, width=480, camera_id=0):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        return self.sim._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

    def close(self):
        pass


class ArchTestCallback(TaskCallback):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.reward = 0.0

    def after_step(self, task, physics):
        self.reward += FarmsGym.compute_reward(task.data.sensors, task.iteration - 1)
        # if task.iteration > 1499:
        #     print(f"iteration: {task.iteration}")
        #     print(f"reward: {self.reward}")


class GymTestCallback(TaskCallback):
    """GymTestCallback callback

    @brief: Testing class; control of the environment is within farms; the TaskCallback is inerited from farms

    """

    def __init__(
        self,
        timestep: float,
        n_iterations: int,
        model,
        observation_choice,
        action_choice,
        **kwargs,
    ):
        super().__init__()
        self.timestep = timestep
        self.n_iterations = n_iterations
        self.model = model  # policy
        self.observations = None
        self.observation_choice = observation_choice
        self.action_choice = action_choice
        self.n_obs = observation_choice.n_obs
        self.n_act = action_choice.n_act

        # default action if model is none
        # self.action = np.zeros(self.n_act) if self.model is None else None
        self.sim = None

        self.debug_random_cond = kwargs.pop("debug_random_cond", True)
        self.notion = kwargs.pop("notion", None)

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        self.observations = FarmsGym.get_observations(
            data_sensors=task.data.sensors,
            data_states=task.data.state,
            iteration=0,
            observation_choice=self.observation_choice,
        )
        return

    def before_step(self, task, action, physics):
        """Take Action based on previous observation"""
        if self.debug_random_cond:
            # skip the action
            return

        if self.model is None:
            raise ValueError("model cannot be none")
        self.action, _states = self.model.predict(self.observations, deterministic=True)

        # sim is mujoco simulation object
        FarmsGym.set_action(
            action=self.action,
            network_parameters=self.sim.task.data.network,
            action_choice=self.action_choice,
            iteration=task.iteration,
        )
        return

    def after_step(self, task, physics):
        """After each step"""
        iteration = task.iteration - 1
        self.observations = FarmsGym.get_observations(
            data_sensors=task.data.sensors,
            data_states=task.data.state,
            iteration=iteration,
            observation_choice=self.observation_choice,
        )

        # TODO calculate reward here

        return

    def set_mujoco_model(self, sim):
        self.sim = sim


if __name__ == "__main__":
    raise ValueError("Not a file that is supposed to be run")
