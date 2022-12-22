"""Simulation"""

from mimetypes import init
import os
from textwrap import wrap
import warnings
import traceback

import numpy as np

from dm_control.rl.control import Environment, PhysicsError
from dm_env import TimeStep, StepType

from farms_core import pylog
from farms_mujoco.simulation.application import FarmsApplication
from farms_mujoco.simulation.task import TaskCallback

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

# from cmc.salamandra_simulation.test import wrap_2pi

class FarmsGym(gym.Env):
    """Simulation"""

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
                [-np.inf]*(n_obs)
                ),
            high=np.array(
                [np.inf]*(n_obs)
                ),
        )

        self.timestep = timestep
        # weights for each sensory feedback
        self.action_space = spaces.Box(
            low=np.array([-1]*n_act),
            high=np.array([1]*n_act),
        )
        self.n_obs = n_obs
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

    def get_observations(data_sensors, data_states, iteration):
        """get observation
        
        AgnathaX: stretch(joint angles) : observation

        """
        joints_pos = np.array(data_sensors.joints.positions(iteration=iteration))
        return joints_pos


    def compute_reward(timestep, data_sensors, data_states, iteration, prev_iteration, debug=False):
        reward = 0
        if prev_iteration < 0:
            return reward
        # reward_phase = .1*FarmsReward.reward_phases(data_states, iteration, debug)
        reward_pc = .01*FarmsReward.reward_phase_lag_const(timestep, data_states, iteration, debug)
        reward_sf = FarmsReward.reward_speed_forward(timestep, data_sensors, iteration, prev_iteration, debug)
        reward_df = FarmsReward.reward_distance_forward(timestep, data_sensors, iteration, 0, debug)
        reward_dft = FarmsReward.reward_distance_forward_tracking(timestep, data_sensors, iteration, 0, debug)
        reward_ct = FarmsReward.reward_contacts_test(timestep, data_sensors, iteration, 0, debug)
        reward_sft = FarmsReward.reward_speed_forward_tracking(timestep, data_sensors, iteration, prev_iteration, debug)
        if debug:
            print('Reward PC        : {}'.format(reward_pc))
            print('Reward SF        : {}'.format(reward_sf))
            print('Reward DF        : {}'.format(reward_df))
            print('Reward DFT       : {}'.format(reward_dft))
            print('Reward CT        : {}'.format(reward_ct))
            print('Reward SFT       : {}'.format(reward_sft))

        return (reward_pc + reward_sf + reward_df + reward_dft + reward_ct + reward_sft)
    
    def set_action(action, robot_parameters, test_type):
        """ Apply the computed action to the concerned variables"""
        action = action*30
        for elem in action:
            if elem is None or elem is np.NaN or abs(elem) is np.inf:
                print("not right")
        
        # action = 1/2 of robot_parameters
        # while i in range(len(action)):
        #     robot_parameters[i] = action_val
        #     robot_parameters[]
        for i,action_val in enumerate(action):
            robot_parameters[i*2+0] = action_val
            robot_parameters[i*2+1] = action_val*-1

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
            # robot_parameters=self.sim.task._controller.network,
            robot_parameters=self.sim.task.data.network.joints2osc_map.weights.array,
            test_type=None,
            )
        env_step = self.sim._env.step(action=None)
        self.observation = FarmsGym.get_observations(
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=iteration)
        
        self.reward = FarmsGym.compute_reward(
            timestep=self.timestep,
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=iteration,
            prev_iteration=iteration - 500)
        end_episode = FarmsGym.arena_limit_reached(
            timestep=self.timestep,
            data_sensors=self.sim.task.data.sensors,
            data_states=self.sim.task.data.state,
            iteration=iteration,
        )
        if end_episode:
            print("episode should be done")
        self.done = True if env_step.step_type == StepType.LAST else False
        return self.observation, self.reward, self.done, self.info
    
    def reset(self):
        self.info = {}
        self.done = False
        self.reward = 0
        self.observation = np.array([0.0]*self.n_obs, dtype=np.float32)
        data_state_copy = np.array(self.sim.task.data.state.array[0,:])
        # self.sim.task.iteration = 0
        self.sim._env.reset()
        self.sim.task.data.sensors.contacts.array[:] = 0.0
        self.sim.task.data.sensors.joints.array[:] = 0.0
        self.sim.task.data.sensors.links.array[:] = 0.0
        self.sim.task.data.sensors.xfrc.array[:] = 0.0
        self.sim.task.data.state.array[:] = 0.0 

        # self.sim.task.data.reset()
        # initial_phases_r = None if self.initial_phase_generator is None else self.initial_phase_generator()
        # if initial_phases_r is None:
        #     raise ValueError("Cannot be none")
        # self.sim.task._controller.network.reset(initial_phases_r)
        self.sim.task.data.state.array[0,:] = data_state_copy
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

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        self.observations = FarmsGym.get_observations(
            data_sensors=task.data.sensors,
            data_states=task.data.state,
            iteration=0,
        )
        return 

    def before_step(self, task, action, physics):
        """Take Action based on previous observation"""
        if self.model is not None:
            self.action, _states = self.model.predict(self.observations )
        # t1 = task.data.network.joints2osc_map.connections
        # t2 = task.data.network.joints2osc_map.weights
        # FarmsGym.set_action(
        #     action=self.action,
        #     # robot_parameters=self.notion,
        #     robot_parameters=task.data.network.joints2osc_map.weights.array,
        #     test_type=None,
        # )
        return

    def after_step(self, task, physics):
        """After each step"""
        iteration = task.iteration -1
        self.observations = FarmsGym.get_observations(
            data_sensors=task.data.sensors,
            data_states=task.data.state,
            iteration=iteration,
        )
        reward = FarmsGym.compute_reward(
            timestep=self.timestep,
            data_sensors=task.data.sensors,
            data_states=task.data.state,
            iteration=iteration,
            prev_iteration=iteration - 100,
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
            print("episode limit........................")
        return
    
    def reset(self):
        """Reset the observations"""
        self.observations = np.array([0.0]*self.n_obs, dtype=np.float32)
        return