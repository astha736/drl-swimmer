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


class ObservationType(Enum):
    JOINT_POSITION = 1
    REACTION_X = 2
    REACTION_Y = 3
    REACTION_Z = 4
    REACTION_XY = 5
    REACTION_XYZ = 6
    PHASES = 7


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

    def observation_bound_PHASES(self):
        """PHASES"""
        low = np.array([-np.inf] * (self.n_body_joints))  # why not +1?
        high = np.array([np.inf] * (self.n_body_joints))  # whoy not +1?
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

        # choose action space according to experiment
        if conf.CONF["RL"]["localFeedback"]:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
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

        self.notion = kwargs.pop("notion", None)

        self.is_test_env = is_test_env

        self.sim, _ = simulation.setup_simulation(
            self.animat_options,
            self.arena_options,
            self.sim_options,
            self.simulator,
            callbacks=[],
        )

        if conf.CONF["RL"]["localFeedback"]:
            self.action_buffer = np.zeros(self.n_act)
            self.action_buffer_counter = 0

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

        # healthy
        healthy = conf.CONF["RL"]["RewardFnc"]["healthy"]

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

        return (
            conf.CONF["RL"]["RewardFnc"]["forward_x"] * forward_x
            + conf.CONF["RL"]["RewardFnc"]["cmd_torques"] * cmd_torques
            + conf.CONF["RL"]["RewardFnc"]["active_torques"] * active_torque
            + healthy
            + conf.CONF["RL"]["RewardFnc"]["forward_com"] * forward_com
            + conf.CONF["RL"]["RewardFnc"]["active_torque_diff"] * active_torque_diff
        )

    def set_action(
        action, network_parameters, action_choice: ActionChoice, iteration: int
    ):
        """Apply the computed action to the concerned variables"""
        if np.isnan(action).any():
            raise ValueError("Action is nan")

        if (action > 1).any() or (action < -1).any():
            raise ValueError("Action is out of bounce")

        if action is None:
            raise ValueError("should not be allowed")

        action_choice.set_action(action, network_parameters, iteration)
        return

    def step(self, action):
        """Performs a step on the environment"""

        # global feedback: step every iteration
        # local feedback: use action buffering
        if conf.CONF["RL"]["localFeedback"]:
            self.action_buffer[self.action_buffer_counter] = action[0]
            self.action_buffer_counter += 1
            # return if we have not yet filled the buffer
            if not self.action_buffer_counter == self.n_act:
                return self.observation, self.reward, self.done, self.info
            # reset action_buffer_counter and perform a step in the environment
            else:
                action = self.action_buffer
                self.action_buffer_counter = 0

        # this MUST be defined here!
        iteration = self.sim.task.iteration
        if action is None:
            print("should not be allowed")

        FarmsGym.set_action(
            action=action,
            network_parameters=self.sim.task.data.network,
            action_choice=self.action_choice,
            iteration=iteration,
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

        if self.done and self.is_test_env:
            utils.save_performance_metrics(
                self.sim,
                self.timestep,
                self.sim_options.n_iterations,
            )
            if self.sim_options.record == True:
                postprocessing_from_clargs(
                    sim=self.sim,
                    video_name=os.path.join(conf.LOG_DIR_RESULTS, "best_model.mp4"),
                )

        return self.observation, self.reward, self.done, self.info

    def randomize_robot_state(self):
        """Randomize the robot state at each rest

        Robot state:
            Spawn: pose and orientation
            Joints: initial position and velocity(default to 0)

        """

        # get new changes (joint and spawn) via animat_options
        # RobotInitialState.set_random_shape_pose(animat_options=self.sim.task.animat_options)
        # RobotInitialOscillator.random_oscillator_phase(animat_options=self.sim.task.animat_options)

        # RobotInitialState.set_random_shape_pose(animat_options=self.sim._env._task.animat_options)
        # RobotInitialOscillator.random_oscillator_phase(animat_options=self.sim._env._task.animat_options)

        # self.sim, _ = simulation.setup_simulation(
        #     self.animat_options,
        #     self.arena_options,
        #     self.sim_options,
        #     self.simulator,
        #     callbacks = []
        # )

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
        # self.randomize_robot_state()
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
        if task.iteration > 1499:
            print(f"iteration: {task.iteration}")
            print(f"reward: {self.reward}")


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
