from asyncore import write
import os
import numpy as np
import pickle
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch
import torch as th
from torch import nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from rlgym.rl_gym import FarmsGym, GymTestCallback, ActionChoice, ObservationChoice, ArchTestCallback

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
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = conf.CONF["RL"]["policy_network"]["arch"][1]
        self.latent_dim_vf = conf.CONF["RL"]["policy_network"]["arch"][1]

        

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_pi),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, conf.CONF["RL"]["policy_network"]["arch"][0]),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
            nn.Linear(conf.CONF["RL"]["policy_network"]["arch"][0], self.latent_dim_vf),
            getattr(torch.nn, conf.CONF["RL"]["policy_network"]["act_fn"])(),
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
        def get_vec_env():
            gym_env = FarmsGym(
                timestep=self.sim_options.timestep,
                observation_choice=self.observation_choice,
                action_choice=self.action_choice,
                animat_options=self.animat_options,
                arena_options = self.arena_options,
                sim_options = self.sim_options,
                simulator = self.simulator,
                log_dir=self.log_dir,
            )
            return gym_env

        vec_gym_env = make_vec_env(
            get_vec_env, n_envs=1, seed=123 # vec_env_cls=SubprocVecEnv
        )

        if conf.CONF["RL"]["PPOparams"]["norm_obs"]:
            vec_gym_env = VecNormalize(vec_gym_env, norm_obs=True, norm_reward=True)

        model = PPO(
            CustomActorCriticPolicy,
            vec_gym_env,
            tensorboard_log=self.log_dir,
            seed=123,
            learning_rate=linear_schedule(conf.CONF["RL"]["PPOparams"]["lr_start"], conf.CONF["RL"]["PPOparams"]["lr_end"]),
            n_steps=conf.CONF["RL"]["PPOparams"]["n_steps"],
            batch_size=conf.CONF["RL"]["PPOparams"]["batch_size"],
            n_epochs=conf.CONF["RL"]["PPOparams"]["n_epochs"],
            gamma=conf.CONF["RL"]["PPOparams"]["gamma"],
            gae_lambda=conf.CONF["RL"]["PPOparams"]["gae_lambda"],
            use_sde=conf.CONF["RL"]["PPOparams"]["use_sde"],
            sde_sample_freq=conf.CONF["RL"]["PPOparams"]["sde_sample_freq"],
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
            callback_on_new_best=SaveVecNormalizeCallback(save_freq=1, name_prefix="best_model", save_path=conf.LOG_DIR) if conf.CONF["RL"]["PPOparams"]["norm_obs"] else None,
        )

        # profile.profile(
       	#     function=model.learn, total_timesteps=1, profile_filename="profile_prod_cluster_2cpu.profile"
        # )

        model.learn(total_timesteps=self.learn_total_timesteps , callback=eval_callback)
        model.save(os.path.join(conf.LOG_DIR, "last_model_trained.zip"))
        if conf.CONF["RL"]["PPOparams"]["norm_obs"]:
            model.get_vec_normalize_env().save(os.path.join(conf.LOG_DIR, "last_model_trained_normalize.pkl"))
        
        ##### TEST #####
        del model, vec_gym_env

        self.sim_options.record = True

        def get_test_vec_env():
            gym_env_test = FarmsGym(
                timestep=self.sim_options.timestep,
                observation_choice=self.observation_choice,
                action_choice=self.action_choice,
                animat_options=self.animat_options,
                arena_options = self.arena_options,
                sim_options = self.sim_options,
                simulator = self.simulator,
                log_dir=self.log_dir,
                is_test_env=True,
            )
            return gym_env_test
        
        vec_gym_env_test = make_vec_env(
            get_test_vec_env, n_envs=1, seed=123
        )

        if conf.CONF["RL"]["PPOparams"]["norm_obs"]:
            vec_gym_env_test = VecNormalize.load(os.path.join(conf.LOG_DIR, "best_model_normalize.pkl"), vec_gym_env_test)
            vec_gym_env_test.training = False
            vec_gym_env_test.norm_reward = False

        model = PPO.load(os.path.join(conf.LOG_DIR, "best_model.zip"))


        from stable_baselines3.common.evaluation import evaluate_policy
        rew, len_ = evaluate_policy(
            model,
            vec_gym_env_test,
            n_eval_episodes=1,
            deterministic=True,
            return_episode_rewards=True,
        )

        # log reward of best model to performance_metrics.txt
        with open(os.path.join(conf.LOG_DIR, "performance_metrics.txt"), "a") as f:
            f.write("\n")
            f.write(f"best model reward: {rew} \n")
        f.close()

        # log reward of best model to common results file: results.yaml
        results_file = "./experiments/results.yaml"
        results = yaml.load((open(results_file, "r")), Loader=yaml.FullLoader)
        results[conf.CONF["experiment_id"]]["best model reward"] = f'{rew}'
        with (open(results_file, "w")) as f:
            f.write(yaml.dump(results))
        f.close()



    # This is another way to test a model; not used for now
    def exp_testing(self, model_filename: str, debug_random_cond: bool) -> None:
        """Experiment testing

        @param model_filename (str): Name of the saved model.
        @param debug_random_cond (bool): If true, the animat is tested in random conditions.
        """
        # load trained model
        model = PPO.load(
            "./experiments/030/logs/06-06-2023_10:05:36/best_model.zip",
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
        archTestCallback = ArchTestCallback()
      
        sim, animat_data = simulation.setup_simulation(
            self.animat_options,
            self.arena_options,
            self.sim_options,
            self.simulator,
            callbacks=[archTestCallback],
        )

        sim._env.reset()
        sim.run()


        # get and save plots and data
        utils.save_performance_metrics(
            sim,
            self.log_dir,
            self.sim_options.timestep,
            self.sim_options.n_iterations,
        )

        # log reward of best model to performance_metrics.txt
        with open(os.path.join(conf.LOG_DIR, "performance_metrics.txt"), "a") as f:
            f.write("\n")
            f.write(f"best model reward: {archTestCallback.reward} \n")
        f.close()

        # log reward of best model to common results file: results.yaml
        results_file = "./experiments/results.yaml"
        results = yaml.load((open(results_file, "r")), Loader=yaml.FullLoader)
        results[conf.CONF["experiment_id"]]["best model reward"] = f'{archTestCallback.reward}'
        with (open(results_file, "w")) as f:
            f.write(yaml.dump(results))
        f.close()


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_normalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                print(f"#### New best model at {self.num_timesteps}")
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True
    
class LinearSchedule():
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.

    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def linear_schedule(initial_value: float, end_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * (initial_value - end_value) + end_value

    return func