"""Simulation"""

from mimetypes import init
import os
from textwrap import wrap
import warnings
import traceback

import numpy as np
import random

from dm_control.rl.control import Environment, PhysicsError
from dm_env import TimeStep, StepType

from farms_core import pylog
from farms_mujoco.simulation.application import FarmsApplication
from farms_mujoco.simulation.task import TaskCallback
from farms_mujoco.simulation.mjcf import euler2mjcquat

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
from ultils.limbless_init_condition import LimblessExperimentRobotState
from ultils.limbless_oscillator import LimblessExperimentOscillator

# from cmc.salamandra_simulation.test import wrap_2pi

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class FarmsGym(gym.Env):
    """Simulation"""

    # exponential filtering for action
    prev_action = None
    action_weight = 0.1
    action_scale = 60

    def __init__(
            self,
            timestep,
            n_obs,
            n_act,
            sim,
            notion,
            **kwargs,
    ):
        super().__init__()
        # stretch feedback
        self.observation_space = spaces.Box(
            low=np.array(
                [-np.inf]*(10)
                + [-np.inf]*(10+1)
                ),
            high=np.array(
                [np.inf]*(10)
                + [np.inf]*(10+1)
                ),
        )

        self.timestep = timestep
        # weights for each sensory feedback
        self.action_space = spaces.Box(
            low=np.array([-1]*n_act),
            high=np.array([1]*n_act),
        )
        self.n_obs = 2*10 + 1
        self.n_act = n_act
        self.sim = sim
        self.init_com_position = np.array(kwargs.pop('init_com_position', None))
        self.init_com_orientation = np.array(kwargs.pop('init_com_orientation', None))
        # self.initial_phase_generator = kwargs.pop('initial_phase_generator', None)
        assert self.init_com_position is not None, "ERROR: init_com_position should be set"
        assert self.init_com_orientation is not None, "ERROR: init_com_orientation should be set"

        self.reward = None
        self.info = None
        self.done = None
        self.observation = None
        self.notion = notion
        self.random_times = 0
        FarmsGym.prev_action = np.zeros(n_act)

    def get_observations(data_sensors, data_states, iteration):
        """get observation

        AgnathaX: Observation space of stretch & contact forces
            stretch(joint angles) : observation
            contact(reaction forces): observation
        """
        joints_pos = np.array(data_sensors.joints.positions(iteration=iteration))
        data_reaction_forces = np.array(data_sensors.contacts.array[iteration,:,0:2])
        isNaN = np.isnan(data_reaction_forces).any()
        if isNaN:
            warnings.warn(bcolors.WARNING + "NaN values in contact forces at itr {}".format(iteration) + bcolors.ENDC)
        np.nan_to_num(data_reaction_forces,copy=False, nan=0.0, posinf=0.0, neginf=-0.0)
        data_reaction_forces_norm = np.linalg.norm(data_reaction_forces, axis=1)
        obs = np.append(joints_pos, data_reaction_forces_norm)
        return obs, isNaN


    def compute_reward(timestep, data_sensors, data_states, iteration, prev_iteration, debug=False):
        reward = 0
        if prev_iteration < 0:
            return reward
        
        # reward_pc = .1*FarmsReward.reward_phase_lag_const(timestep, data_states, iteration, debug)
        # reward_df = FarmsReward.reward_distance_forward(timestep, data_sensors, iteration, prev_iteration, debug)
        reward_dft = FarmsReward.reward_distance_forward_tracking(timestep, data_sensors, iteration, 0, debug)
        reward_ct = FarmsReward.reward_contacts_test(timestep, data_sensors, iteration, 0, debug)
        prev_iteration_speed = (iteration - int(0.5/timestep))
        # reward_sf = FarmsReward.reward_speed_forward(timestep, data_sensors, iteration, prev_iteration_speed, debug)
        # reward_cot = 3*FarmsReward.cost_of_transport(timestep, data_sensors, iteration, prev_iteration_speed, debug)
        # reward_sft = 3*FarmsReward.reward_speed_forward_tracking(timestep, data_sensors, iteration, prev_iteration_speed, debug)
        # r_sum = (reward_pc + reward_sf + reward_df + reward_dft + reward_ct + reward_sft + reward_cot)
        r_sum = reward_dft + reward_ct
        if debug:
            # print('Reward PC        : {}'.format(reward_pc))
            # print('Reward DF        : {}'.format(reward_df))
            print('Reward DFT       : {}'.format(reward_dft))
            print('Reward CT        : {}'.format(reward_ct))
            # print('Reward Speed F   : {}'.format(reward_sf))
            # print('Reward Speed FT  : {}'.format(reward_sft))
            # print('Reward COT       : {}'.format(reward_cot))
            print('SUM************  : {}'.format(r_sum))

        return r_sum
    
    def set_action(action, robot_parameters, test_type):
        """ Apply the computed action to the concerned variables"""
        isNaN = np.isnan(action).any()
        if isNaN:
            warnings.warn(bcolors.WARNING + "NaN values in action" + bcolors.ENDC)
        # np.nan_to_num(action, copy=False, nan=0.0, posinf=0.0, neginf=-0.0)
        if (action > 1).any() or (action < -1).any():
            warnings.warn(bcolors.WARNING + "NaN action values not in range" + bcolors.ENDC)
        
        action_curr = FarmsGym.action_weight*(action) + (1 - FarmsGym.action_weight)*FarmsGym.prev_action
        action_curr = action_curr*FarmsGym.action_scale

        if (action_curr > FarmsGym.action_scale).any() or (action < -FarmsGym.action_scale).any():
            warnings.warn(bcolors.WARNING + "weights value not in range" + bcolors.ENDC)

        for elem in action_curr:
            if elem is None or elem is np.NaN or abs(elem) is np.inf:
                print("not right")
         
        for i,action_val in enumerate(action_curr):
            robot_parameters[i*2+0] = action_val    # left oscillator assignment 
            robot_parameters[i*2+1] = action_val*-1 # right oscillator assignment
        
        FarmsGym.prev_action = action
        return
    
    def arena_limit_reached(timestep, data_sensors, data_states, iteration, debug=False):
        com_position = np.array(
            data_sensors.links.com_position(
            iteration=iteration,
            link_i=0,
            )
        )
        x_limit = com_position[0] > 3 or  com_position[0] < -1
        y_limit = np.abs(com_position[1]) > 2

        limit_reached = x_limit or y_limit
        if debug:
            print("[episode info] COM   : {}".format(com_position[0:2]))
            print("[episode info] limit : {}".format(limit_reached))
        return limit_reached

    def step(self, action):
        """Given the action change the control command ?"""

        # iteration changes after the env step
        iteration = self.sim.task.iteration
        if action is None:
            print("should not be allowed")
        
        FarmsGym.set_action(
            action=action,
            robot_parameters=self.sim.task.data.network.joints2osc_map.weights.array,
            test_type=None,
            )
        env_step = self.sim._env.step(action=None)
        self.observation, isNaN = FarmsGym.get_observations(
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=iteration)
        
        self.reward = FarmsGym.compute_reward(
            timestep=self.timestep,
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=iteration,
            prev_iteration=(iteration - int(1/self.timestep)),
            )
        end_episode = FarmsGym.arena_limit_reached(
            timestep=self.timestep,
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=iteration,
        )
        if end_episode:
            print("episode should be done")
        self.done = True if (env_step.step_type == StepType.LAST) or end_episode else False
        return self.observation, self.reward, self.done, self.info
    
    def randomize_robot_state(self):
        """Randomize the robot state at each rest

        Robot state:
            Spawn: pose and orientation 
            Joints: initial position and velocity(default to 0)
            
        """

        animat_options = self.sim.task.animat_options 

        # get new changes (joint and spawn) via animat_options
        # LimblessExperimentRobotState.set_random_shape_pose(animat_options=animat_options)
        # LimblessExperimentOscillator.random_oscillator_phase(animat_options=animat_options)

        self.random_times =+ 1
        # apply spawn changes
        base_link = self.sim._mjcf_model.worldbody.body[-1]
        base_link.pos = [pos for pos in  animat_options.spawn.pose[:3]]
        base_link.quat = euler2mjcquat( animat_options.spawn.pose[3:])

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

        self.observation, isNan = FarmsGym.get_observations(
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=0)
        
        # for internal use? 
        self.info = {}
        self.done = False
        self.reward = 0

        return self.observation  # reward, done, info can't be included
    
    def render(self, mode='rgb_array', height=480, width=480, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        return self.sim._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

    def close (self):
        pass

class GymTestCallback(TaskCallback):
    """GymTestCallback callback"""

    def __init__(
            self,
            timestep: float,
            n_iterations: int,
            model,
            notion,
            **kwargs,
    ):
        super().__init__()
        self.timestep = timestep
        self.n_iterations = n_iterations
        self.model = model
        self.observations = None
        self.n_obs  = kwargs.pop('n_obs')
        self.n_act  = kwargs.pop('n_act')
        header_obs = ['n_obs_{}'.format(i) for i in range(self.n_obs)]
        header_act = ['n_act_{}'.format(i) for i in range(self.n_act)]
        # default action if model is none
        self.action = np.zeros(self.n_act) if self.model is None else None
        self.notion = notion
        self.sim = None
        self.debug_random_cond = kwargs.pop('debug_random_cond', True)
        FarmsGym.prev_action = np.zeros(self.n_act)

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        self.observations, isNan = FarmsGym.get_observations(
            data_sensors=task.data.sensors,
            data_states=task.data.state,
            iteration=0,
        )
        return 

    def before_step(self, task, action, physics):
        """Take Action based on previous observation"""
        if self.debug_random_cond:
            # skip the action
            return
        else:
            if self.model is None:
                raise ValueError("model cannot be none")
            self.action, _states = self.model.predict(self.observations )
            FarmsGym.set_action(
                action=self.action,
                robot_parameters=task.data.network.joints2osc_map.weights.array,
                test_type=None,
            )
        return

    def after_step(self, task, physics):
        """After each step"""
        iteration = task.iteration -1
        self.observations, isNaN = FarmsGym.get_observations(
            data_sensors=task.data.sensors,
            data_states=task.data.state,
            iteration=iteration,
        )
        reward = FarmsGym.compute_reward(
            timestep=self.timestep,
            data_sensors=task.data.sensors,
            data_states=task.data.state,
            iteration=iteration,
            prev_iteration=(iteration - int(1/self.timestep)),
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
            LimblessExperimentRobotState.set_random_shape_pose(animat_options=animat_options)
            LimblessExperimentOscillator.random_oscillator_phase(animat_options=animat_options)
            # apply spawn changes
            base_link = self.sim._mjcf_model.worldbody.body[-1]
            base_link.pos = [pos for pos in animat_options.spawn.pose[:3]]
            base_link.quat = euler2mjcquat(animat_options.spawn.pose[3:])
            # apply motor pose & oscillator changes with env's reset
            self.sim._env.reset()

            # only for visuals 
            self.sim.task._app._restart_runtime()
            self.sim.task._app._perform_deferred_reload()

        self.observation, isNaN = FarmsGym.get_observations(
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=0)

        return self.observation  # reward, done, info can't be included

    def set_mujoco_model(self, sim):
        self.sim = sim

if __name__ == '__main__':
    raise ValueError("Not a file that is supposed to be run")