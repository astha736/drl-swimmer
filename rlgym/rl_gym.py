"""Simulation"""

from mimetypes import init
import os
from textwrap import wrap
import warnings
import traceback
from enum import Enum

import numpy as np
import random

from dm_control.rl.control import Environment, PhysicsError
from dm_env import TimeStep, StepType

from farms_core import pylog
from farms_mujoco.simulation.application import FarmsApplication
from farms_mujoco.simulation.task import TaskCallback
from farms_mujoco.simulation.mjcf import euler2mjcquat
from typing import List

import gym
from gym import spaces
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

import csv

from .rl_reward import FarmsReward
from utils.limbless_spawn import RobotInitialState
from utils.limbless_oscillator import RobotInitialOscillator

# from cmc.salamandra_simulation.test import wrap_2pi


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


class ObservationType(Enum):
    JOINT_POSITION = 1
    REACTION_X = 2
    REACTION_Y = 3
    REACTION_Z = 4
    REACTION_XY = 5
    REACTION_XYZ = 6


class ActionChoice:
    """_summary_

    Keeping the action lower and upper values between -1 and 1
    """

    action_output_scale = {
        ActionType.STRETCH: 30,
        ActionType.CONTACT: 10,
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
        self.action_length[ActionType.STRETCH] = self.n_body_joints - 1
        low = np.array([-1] * self.action_length[ActionType.STRETCH])
        high = np.array([1] * self.action_length[ActionType.STRETCH])
        return low, high

    def action_bound_CONTACT(self):
        """Contacts are usually same as joint"""

        self.action_length[ActionType.CONTACT] = self.n_body_joints
        low = np.array([-1] * self.action_length[ActionType.CONTACT])
        high = np.array([1] * self.action_length[ActionType.CONTACT])

        return low, high

    def get_action_bound(self, action: ActionType):
        switcher = {
            ActionType.STRETCH: self.action_bound_STRETCH,
            ActionType.CONTACT: self.action_bound_CONTACT,
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

    def set_action_STRETCH(self, action, network_parameters, iteration):
        # @ASTHA PPO OUTPUTS ACTION IN [-1,1]?
        # @ASTHA: WHERE DOES SCALING FACTORS COME FROM?
        action_old = action
        action = action * ActionChoice.action_output_scale[ActionType.STRETCH]
        # @ASTHA: WHAT DOES FOLLOWIGN LINE DO?
        robot_parameters = network_parameters.joints2osc_map.weights.array

        for i, action_val in enumerate(action):
            # LOG PRINT OBSERVE
            robot_parameters[i * 2 + 0] = action_val  # left oscillator assignment
            robot_parameters[i * 2 + 1] = action_val * -1  # right oscillator assignment
        pass

    def set_action_CONTACT(self, action, network_parameters, iteration):

        action = action * ActionChoice.action_output_scale[ActionType.STRETCH]

        # ASTHA BUG FIX
        robot_parameters = network_parameters.contact2osc_map.weights.array

        for i, action_val in enumerate(action):
            robot_parameters[i * 2 + 0] = action_val  # left oscillator assignment
            robot_parameters[i * 2 + 1] = action_val * -1  # right oscillator assignment

        pass

    def set_action_switch(self, observation: ActionType):
        # that a simulated switch case with a dict. Basically, the corresponding action in the
        # dict is called
        switcher = {
            ActionType.STRETCH: self.set_action_STRETCH,
            ActionType.CONTACT: self.set_action_CONTACT,
        }

        return switcher.get(observation, "Invalid observation Type")

    def set_action(self, actions, network_parameters, iteration: int):
        index = 0
        for action_type in self.action_list:
            action_len = self.action_length[action_type]
            action_slice = actions[index : index + action_len]
            self.set_action_switch(action_type)(
                action_slice, network_parameters, iteration
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

    def get_observation_bound(self, observation: ObservationType):
        switcher = {
            ObservationType.JOINT_POSITION: self.observation_bound_JOINT_POSITION,
            ObservationType.REACTION_X: self.observation_bound_REACTION_X,
            ObservationType.REACTION_Y: self.observation_bound_REACTION_Y,
            ObservationType.REACTION_Z: self.observation_bound_REACTION_Z,
            ObservationType.REACTION_XY: self.observation_bound_REACTION_XY,
            ObservationType.REACTION_XYZ: self.observation_bound_REACTION_XYZ,
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

    def extract_observation_JOINT_POSITION(self, data_sensors, iteration):

        joints_pos = np.array(data_sensors.joints.positions(iteration=iteration))

        return joints_pos

    def extract_observation_REACTION_X(self, data_sensors, iteration):
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

    def extract_observation_REACTION_Y(self, data_sensors, iteration):
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

    def extract_observation_REACTION_Z(self, data_sensors, iteration):
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

    def extract_observation_REACTION_XY_NORM(self, data_sensors, iteration):
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

    def extract_observation_REACTION_XYZ_NORM(self, data_sensors, iteration):
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
        }

        return switcher.get(observation, "Invalid observation Type")

    def get_observation(self, data_sensors, iteration: int):

        observations_list = []
        for observation in self.observation_list:
            observation_val = self.extract_observation(observation)(
                data_sensors, iteration
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

    # exponential filtering for action
    prev_action = None
    action_weight = 0.1
    action_scale = 60

    def __init__(
        self,
        timestep,
        observation_choice: ObservationChoice,
        action_choice: ActionChoice,
        sim,
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
        self.sim = sim  # sim contains the farms simulation object, e.g. the agent @ASTHA is this correct?
        # self.initial_phase_generator = kwargs.pop('initial_phase_generator', None)

        # Old code?
        # self.init_com_position = np.array(kwargs.pop('init_com_position', None))
        # self.init_com_orientation = np.array(kwargs.pop('init_com_orientation', None))
        # assert self.init_com_position is not None, "ERROR: init_com_position should be set"
        # assert self.init_com_orientation is not None, "ERROR: init_com_orientation should be set"

        self.reward = None
        self.info = None
        self.done = None
        self.observation = None
        self.random_times = 0

        self.notion = kwargs.pop("notion", None)

        FarmsGym.prev_action = np.zeros(self.n_act)

    def get_observations(
        data_sensors, data_states, iteration: int, observation_choice: ObservationChoice
    ):
        """get observation

        AgnathaX: Observation space is given by the observation_choice (ObservationChoice) which contains a list of
        observations for the given experiment

        """
        return observation_choice.get_observation(
            data_sensors=data_sensors, iteration=iteration
        )

    def compute_reward(
        timestep, data_sensors, data_states, iteration, prev_iteration, debug=False
    ):
        # @ASTHA: data_sensor vs data_states?
        """TODO: cleanup reward"""
        reward = 0
        if prev_iteration < 0:
            return reward

        # reward_pc = FarmsReward.reward_phase_lag_const(timestep, data_states, iteration, debug)
        # reward_df = FarmsReward.reward_distance_forward(
        #     timestep, data_sensors, iteration, prev_iteration, debug
        # )
        reward_dft = FarmsReward.reward_distance_forward_tracking(
            timestep, data_sensors, iteration, 0, debug
        )
        reward_ct = FarmsReward.reward_contacts_test(
            timestep, data_sensors, iteration, 0, debug
        )
        prev_iteration_speed = iteration - int(0.5 / timestep)
        # reward_sf = FarmsReward.reward_speed_forward(timestep, data_sensors, iteration, prev_iteration_speed, debug)
        reward_cot = 3 * FarmsReward.cost_of_transport(
            timestep, data_sensors, iteration, prev_iteration_speed, debug
        )
        # reward_sft = 3*FarmsReward.reward_speed_forward_tracking(timestep, data_sensors, iteration, prev_iteration_speed, debug)
        # r_sum = (reward_pc + reward_sf + reward_df + reward_dft + reward_ct + reward_sft + reward_cot)
        r_sum = reward_dft + reward_ct + reward_cot
        if debug:
            # print('Reward PC        : {}'.format(reward_pc))
            # print('Reward DF        : {}'.format(reward_df))
            print("Reward DFT       : {}".format(reward_dft))
            print("Reward CT        : {}".format(reward_ct))
            # print('Reward Speed F   : {}'.format(reward_sf))
            # print('Reward Speed FT  : {}'.format(reward_sft))
            print("Reward COT       : {}".format(reward_cot))
            print("SUM************  : {}".format(r_sum))

        return r_sum

    def set_action(
        action, network_parameters, action_choice: ActionChoice, iteration: int
    ):
        """Apply the computed action to the concerned variables"""
        isNaN = np.isnan(action).any()
        if isNaN:
            warnings.warn(bcolors.WARNING + "NaN values in action" + bcolors.ENDC)
        # np.nan_to_num(action, copy=False, nan=0.0, posinf=0.0, neginf=-0.0)
        # print(type(action))
        if (action > 1).any() or (action < -1).any():
            warnings.warn(
                bcolors.WARNING + "NaN action values not in range" + bcolors.ENDC
            )

        # @ASTHA RANDOM RESCALING?
<<<<<<< Updated upstream
        # @ASTHA: What is the max reward for the learning?
=======
>>>>>>> Stashed changes
        action_curr = (
            FarmsGym.action_weight * (action)
            + (1 - FarmsGym.action_weight) * FarmsGym.prev_action
        )
        FarmsGym.prev_action = action

        action_choice.set_action(action_curr, network_parameters, iteration)
        return

    def arena_limit_reached(
        timestep, data_sensors, data_states, iteration, debug=False
    ):
        com_position = np.array(
            data_sensors.links.com_position(
                iteration=iteration,
                link_i=0,
            )
        )
        x_limit = com_position[0] > 3 or com_position[0] < -1
        y_limit = np.abs(com_position[1]) > 2

        limit_reached = x_limit or y_limit
        if debug:
            print("[episode info] COM   : {}".format(com_position[0:2]))
            print("[episode info] limit : {}".format(limit_reached))
        return limit_reached

    def step(self, action):
        """Performs a step on the environment"""

        # iteration changes after the env step
        iteration = self.sim.task.iteration
        if action is None:
            print("should not be allowed")

        FarmsGym.set_action(
            action=action,
            network_parameters=self.sim.task.data.network,
            action_choice=self.action_choice,
            iteration=iteration,
        )

        # @ASTHA what does the following do?
        env_step = self.sim._env.step(
            action=None
        )  # Take control of the env; used instead of sim.run
        self.observation = FarmsGym.get_observations(
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=iteration,
            observation_choice=self.observation_choice,
        )

        self.reward = FarmsGym.compute_reward(
            timestep=self.timestep,
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=iteration,
            prev_iteration=(iteration - int(1 / self.timestep)),
        )

        # Add Termination criteria here
        end_episode = FarmsGym.arena_limit_reached(
            timestep=self.timestep,
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=iteration,
        )

        if end_episode:
            print("episode should be done")
        self.done = (
            True if (env_step.step_type == StepType.LAST) or end_episode else False
        )
        return self.observation, self.reward, self.done, self.info

    def randomize_robot_state(self):
        """Randomize the robot state at each rest

        Robot state:
            Spawn: pose and orientation
            Joints: initial position and velocity(default to 0)

        """

        animat_options = self.sim.task.animat_options

        # get new changes (joint and spawn) via animat_options
        # RobotInitialState.set_random_shape_pose(animat_options=animat_options)
        # RobotInitialOscillator.random_oscillator_phase(animat_options=animat_options)

        self.random_times = +1
        # apply spawn changes
        base_link = self.sim._mjcf_model.worldbody.body[-1]
        base_link.pos = [pos for pos in animat_options.spawn.pose[:3]]
        base_link.quat = euler2mjcquat(animat_options.spawn.pose[3:])

        return

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
        # reset the variables for robot state
        self.randomize_robot_state()
        # apply motor pos & oscillator changes along with reset
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

        return self.observation  # reward, done, info can't be included

    def render(self, mode="rgb_array", height=480, width=480, camera_id=0):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        return self.sim._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

    def close(self):
        pass


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

        header_obs = ["n_obs_{}".format(i) for i in range(self.n_obs)]
        header_act = ["n_act_{}".format(i) for i in range(self.n_act)]
        # default action if model is none
        # self.action = np.zeros(self.n_act) if self.model is None else None
        self.sim = None

        self.debug_random_cond = kwargs.pop("debug_random_cond", True)
        self.notion = kwargs.pop("notion", None)
        FarmsGym.prev_action = np.zeros(self.n_act)

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
        self.action, _states = self.model.predict(self.observations)

        pylog.debug("observations: {}".format(self.observations))
        pylog.debug("action: {}".format(self.action))

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
        reward = FarmsGym.compute_reward(
            timestep=self.timestep,
            data_sensors=task.data.sensors,
            data_states=task.data.state,
            iteration=iteration,
            prev_iteration=(iteration - int(1 / self.timestep)),
            debug=True,
        )
        episode_limit = FarmsGym.arena_limit_reached(
            timestep=self.timestep,
            data_sensors=task.data.sensors,
            data_states=task.data.state,
            iteration=iteration,
            debug=True,
        )
        if episode_limit:
            self.reset()
        if self.debug_random_cond and iteration > 1000:
            self.reset()
        return

    def reset(self):
        """Reset the observations"""
        self.info = {}
        self.done = False
        self.reward = 0
        if self.debug_random_cond:
            animat_options = self.sim.task.animat_options
            RobotInitialState.set_random_shape_pose(animat_options=animat_options)
            RobotInitialOscillator.random_oscillator_phase(
                animat_options=animat_options
            )
            # apply spawn changes
            base_link = self.sim._mjcf_model.worldbody.body[-1]
            base_link.pos = [pos for pos in animat_options.spawn.pose[:3]]
            base_link.quat = euler2mjcquat(animat_options.spawn.pose[3:])
            # apply motor pose & oscillator changes with env's reset
            self.sim._env.reset()

            # only for visuals
            self.sim.task._app._restart_runtime()
            self.sim.task._app._perform_deferred_reload()

        self.observation = FarmsGym.get_observations(
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=0,
            observation_choice=self.observation_choice,
        )

        return self.observation  # reward, done, info can't be included

    def set_mujoco_model(self, sim):
        self.sim = sim


if __name__ == "__main__":
    raise ValueError("Not a file that is supposed to be run")
