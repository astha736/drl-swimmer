from asyncore import write
import os
import numpy as np
import pickle
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch
import torch as th
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from rlgym.rl_gym import FarmsGym, GymTestCallback, ActionChoice, ObservationChoice

from . import simulation

from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

from farms_sim.simulation import postprocessing_from_clargs
from farms_amphibious.data.data import AmphibiousData
from farms_core.utils import profile
from . import utils

from gym import spaces

import conf

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
        last_layer_dim_pi: int = 200, # actor=policy
        last_layer_dim_vf: int = 200, # critic
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 200),
            nn.ReLU(),
            nn.Linear(200, last_layer_dim_pi),
            nn.ReLU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 200),
            nn.ReLU(),
            nn.Linear(200, last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

        Customized for very local feedback and shared net
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # Thats wrong. Action space is one dimensional! (Same probability distributions for all actions)
        # Make action space 1D
        # not sure how to handle obs space yet
        # feature = th.Tensor([th.Tensor([features[0][1]])])
        # out = self.policy_net(feature)
        # return th.Tensor([out, out, out, out, out, out, out, out, out, out])

        return self.policy_net(features)

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
        # self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


class TrainTestClass:
    """TrainTestClass

    @brief: TrainTestClass is a class that helps in training, testing and continuing training of RL models.It is designed to be used with the
    FarmsGym environment for training.

    @details: TrainTestClass is a wrapper around the Stable Baselines3 library and OpenAI Gym. It brings together all the components required for
    training, testing and continuing training of RL models. The usual flow of the TrainTestClass is as follows:
        - Create the TrainTestClass object with options for FARMS, Gym Environment, RL model
        - Call the exp_training() method to train the model. This method in turn:
            - creates a simulation object with mujoco through FARMS options
            - creates the FarmsGym environment(Gym Environment) with the simulation object
            - creates the RL model using Stable Baselines3 library
            - trains the model using the FarmsGym environment
        - Call the exp_train_cont() method to continue training the model if needed
        - Call the exp_testing() method to test the model
            - Loads the trained model
            - Creates callback from GymTestCallback class. This callback contains the model and is used to query for action via callbacks.
            - Creates the FARMS simulation with mujoco and provides the above callback

    Training is done with help of the FarmsGym class.
    This class is an openAI Gym environment that uses FARMS as the backend simulator framework. As OpenAI gym requires control for running each
    step of the simulation, the TrainTestClass is used to create the environment and simulation(mujoco through farms) and provide the control to the
    learning environment.

    Testing is performed via GymTestCallback. For testing we use FARMS in its default mode. The GymTestCallback is provided as the callback
    class to the FarmsGym environment. The callback is called at each step. At every step the observation is passed to the model and the action
    given by the model is applied.

    """

    def __init__(
        self,
        animat_options,
        arena_options,
        sim_options,
        simulator,
        log_dir,
        action_choice: ActionChoice,
        observation_choice: ObservationChoice,
        learn_total_timesteps: int,
        clargs=None,
    ):
        """Constructor for TrainTestClass

        Responsibility: train, continue training and testing

        Args:
            animat_options (_type_): animat_options
            arena_options (_type_): arena_options
            sim_options (_type_): simulation options
            simulator (_type_): simulator choice
        """
        self.animat_options = animat_options
        self.arena_options = arena_options
        self.sim_options = sim_options
        self.simulator = simulator
        self.log_dir = log_dir
        self.save_test_data = False  # TODO: setup kwargs option
        self.action_choice = action_choice
        self.observation_choice = observation_choice
        self.learn_total_timesteps = learn_total_timesteps
        self.clargs = clargs

    # Train and test
    def exp_training(self, model_filename: str) -> None:
        """Experiment training

        @param model_filename (str): Name of the saved model.
        """

        ##### TRAIN #####

        # setup simulation
        sim, animat_data = simulation.setup_simulation(
            self.animat_options,
            self.arena_options,
            self.sim_options,
            self.simulator,
            callbacks=[],
        )

        def get_vec_env():
            gym_env = FarmsGym(
                timestep=self.sim_options.timestep,
                observation_choice=self.observation_choice,
                action_choice=self.action_choice,
                sim=sim,
                log_dir=self.log_dir,
            )
            return gym_env

        vec_gym_env = make_vec_env(
            get_vec_env, n_envs=1
        )  # SubprocVecEnv not working due to cython pickling error

        # policy_kwargs = dict(
        #     activation_fn=getattr(
        #         torch.nn, conf.CONF["RL"]["policy_network"]["activation"]
        #     ),
        #     net_arch=dict(
        #         pi=conf.CONF["RL"]["policy_network"]["arch"],  # actor
        #         vf=conf.CONF["RL"]["policy_network"]["arch"],  # critic
        #     ),
        # )

        model = PPO(
            CustomActorCriticPolicy, # conf.CONF["RL"]["policy_network"]["policy_type"],
            vec_gym_env,
            # policy_kwargs=policy_kwargs,
            tensorboard_log=self.log_dir,
        )

        # configure logger
        new_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        eval_callback = EvalCallback(
            vec_gym_env,
            eval_freq=10000,
            deterministic=True,
            warn=True,
            verbose=1,
            # log_path=conf.LOG_DIR, # don't know how to read the log and what's in there
            best_model_save_path=conf.LOG_DIR,
        )

        # profile.profile(
        #     function=model.learn, total_timesteps=1, profile_filename="profile_1_step.txt"
        # )

        # profile.profile(
        #     function=model.learn, total_timesteps=10_000, profile_filename="profile_10000_step.txt"
        # )

        model.learn(total_timesteps=self.learn_total_timesteps , callback=eval_callback)
        model.save(os.path.join(conf.LOG_DIR, "trained_model_last.zip"))

        ##### TEST #####

        # test model once and save performance metrics
        del model
        model = PPO.load("experiments/999/logs/31-05-2023_17:03:36/best_model.zip")

        self.sim_options.record = True
        sim, animat_data = simulation.setup_simulation(
            self.animat_options,
            self.arena_options,
            self.sim_options,
            self.simulator,
            callbacks=[],
        )
        def get_test_vec_env():
            gym_env_test = FarmsGym(
                timestep=self.sim_options.timestep,
                observation_choice=self.observation_choice,
                action_choice=self.action_choice,
                sim=sim,
                log_dir=self.log_dir,
                is_test_env=True,
            )
            return gym_env_test
        
        vec_gym_env_test = make_vec_env(
            get_test_vec_env, n_envs=1
        )  # SubprocVecEnv not working due to cython pickling error
        
        from stable_baselines3.common.evaluation import evaluate_policy
        rew, len_ = evaluate_policy(
            model,
            vec_gym_env_test,
            n_eval_episodes=1,
            deterministic=True,
            return_episode_rewards=True,
        )

        # log reward of best model
        with open(os.path.join(conf.LOG_DIR, "performance_metrics.txt"), "a") as f:
            f.write("\n")
            f.write(f"best model reward: {rew} \n")
        f.close()

    # This is another way to test a model; not used for now
    def exp_testing(self, model_filename: str, debug_random_cond: bool) -> None:
        """Experiment testing

        @param model_filename (str): Name of the saved model.
        @param debug_random_cond (bool): If true, the animat is tested in random conditions.
        """
        # load trained model
        model = PPO.load(
            "experiments/999/logs/31-05-2023_17:03:36/best_model.zip",
        )

        # callback on trained model for testing
        gymTestCallback = GymTestCallback(
            timestep=self.sim_options.timestep,
            n_iterations=self.sim_options.n_iterations,
            model=model,
            observation_choice=self.observation_choice,
            action_choice=self.action_choice,
            debug_random_cond=False,
        )

        sim, animat_data = simulation.setup_simulation(
            self.animat_options,
            self.arena_options,
            self.sim_options,
            self.simulator,
            callbacks=[gymTestCallback],
        )

        gymTestCallback.set_mujoco_model(sim)

        sim.run()

        utils.save_performance_metrics(
            sim,
            self.log_dir,
            self.sim_options.timestep,
            self.sim_options.n_iterations,
        )

    # Testing a CPG-config without a trained model, i.e. analytical
    def arch_testing(self) -> None:
        """Test the architecture of farms

        @brief: This function is used to check the options for FARMS and Notions(ExperimentOptions)
        """
        callbacks = []
        sim, animat_data = simulation.setup_simulation(
            self.animat_options,
            self.arena_options,
            self.sim_options,
            self.simulator,
            callbacks=callbacks,
        )
        # profile.profile(function=sim.run, profile_filename="profile.txt")
        # return

        sim._env.reset()

        sim.run()

        # postprocessing_from_clargs(
        #     sim=sim,
        #     clargs=self.clargs,
        #     simulator=self.simulator,
        #     animat_data_loader=AmphibiousData,
        #     video_name="name.mp4",
        # )

        # writer = SummaryWriter(log_dir=self.log_dir)

        # get and save plots and data
        utils.save_performance_metrics(
            sim,
            self.log_dir,
            self.sim_options.timestep,
            self.sim_options.n_iterations,
        )