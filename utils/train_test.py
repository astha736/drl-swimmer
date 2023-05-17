import os
import numpy as np
import pickle
from enum import Enum
from matplotlib.backends.backend_pdf import PdfPages

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

import torch
from farms_sim.simulation import postprocessing_from_clargs
from farms_amphibious.data.data import AmphibiousData


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
        experiment_args,
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
        self.experiment_args = experiment_args
        self.clargs = clargs

    def exp_training(self, model_filename: str) -> None:
        """Experiment training

        @param model_filename (str): Name of the saved model.
        """

        # setup simulation
        sim, animat_data = simulation.setup_simulation(
            self.animat_options,
            self.arena_options,
            self.sim_options,
            self.simulator,
            callbacks=[],
        )

        # setup gym: create the environment with mujoco sim
        gym_env = FarmsGym(
            timestep=self.sim_options.timestep,
            observation_choice=self.observation_choice,
            action_choice=self.action_choice,
            sim=sim,
        )

        # @ASTHA this throws error
        # check_env(gym_env, warn=True)

        policy_kwargs = dict(
            activation_fn=getattr(
                torch.nn, self.experiment_args["RL"]["policy_network"]["activation"]
            ),
            net_arch=self.experiment_args["RL"]["policy_network"]["arch"],
        )

        model = PPO(
            self.experiment_args["RL"]["policy_network"]["policy_type"],
            gym_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.log_dir,
        )

        # configure logger
        new_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        # train
        eval_callback = EvalCallback(
            gym_env,
            log_path="./logs/",
            eval_freq=500,
            deterministic=True,
            render=False,
            callback_after_eval=evaluteWithFiguresCB(),
        )
        model.learn(
            total_timesteps=self.learn_total_timesteps,
        )
        model.save(os.path.join(str(self.log_dir), str(model_filename)))

    def exp_testing(self, model_filename: str, debug_random_cond: bool) -> None:
        """Experiment testing

        @param model_filename (str): Name of the saved model.
        @param debug_random_cond (bool): If true, the animat is tested in random conditions.
        """
        # load trained model
        model = PPO.load(
            "logs/experiment_01/10-05-2023_22:15:36/sRobotFeedbackSenstivity.NONPERIOD_sW0_caW10_sCaudal_ncCPG/rl_model_1150000_steps.zip"
        )

        # callback on trained model for testing
        gymTestCallback = GymTestCallback(
            timestep=self.sim_options.timestep,
            n_iterations=self.sim_options.n_iterations,
            model=model,
            observation_choice=self.observation_choice,
            action_choice=self.action_choice,
            debug_random_cond=debug_random_cond,
        )
        callbacks = [gymTestCallback]

        # setup simulation
        sim, animat_data = simulation.setup_simulation(
            self.animat_options,
            self.arena_options,
            self.sim_options,
            self.simulator,
            callbacks=callbacks,
        )
        # setup callback for testing model
        gymTestCallback.set_mujoco_model(sim)
        # run simulation
        sim.run()

        if self.save_test_data:
            self.exp_save_run(sim, animat_data)
        return

    def exp_save_run(self, sim, animat_data) -> None:
        """Save simulation data

        @param sim (_type_): simulation object
        @param animat_data (_type_): animat data object
        """
        filename = "{}/test_simulation.{}"
        animat_data.to_file(filename.format(self.log_dir, "h5"), sim.iteration)
        with open(filename.format(self.log_dir, "pickle"), "wb") as param_file:
            pickle.dump(self.sim_options, param_file)
        print(filename.format(self.log_dir, "h5"))
        print(filename.format(self.log_dir, "pickle"))
        return

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
        sim.run()

        # get and save plots and data
        # TODO fix show time and not timesteps in plot
        _times = range(0, self.sim_options.n_iterations - 1)
        plots = {
            **sim.task.data.sensors.links.plots(times=_times),
            **sim.task.data.sensors.joints.plots(times=_times),
        }
        with PdfPages(os.path.join(self.log_dir, "performance_plots.pdf")) as pdf:
            for name, plot in plots.items():
                plot.suptitle(name.replace("_", " "))
                pdf.savefig(plot.figure, bbox_inches="tight")

        metrics = {
            **sim.task.data.sensors.links.performance_metrics(),
            **sim.task.data.sensors.joints.performance_metrics(),
        }
        f = open(os.path.join(self.log_dir, "performance_metrics.txt"), "w")
        for name, metric in metrics.items():
            f.write(f"{name}: {metric}\n")
        f.close()

        return


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)

        env = self.model.env.envs[0].env

        self.logger.record(
            "test_step",
            env.sim.task.data.sensors.links.com_distance_travelled_in_axis(),
        )
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        env = self.model.env.envs[0].env

        self.logger.record(
            "test_rollout",
            env.sim.task.data.sensors.links.com_distance_travelled_in_axis(),
        )
        pass


class evaluteWithFiguresCB(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        print("in here")
