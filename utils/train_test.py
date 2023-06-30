import os
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import yaml
import matplotlib.pyplot as plt
import imageio
import glob
import time

from rlgym.rl_gym import (
    FarmsGym,
    GymTestCallback,
    ActionChoice,
    ObservationChoice,
    ArchTestCallback,
)

from . import simulation

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

from farms_sim.simulation import postprocessing_from_clargs
from farms_core.utils import profile
from . import utils

from utils.networks import CustomActorCriticPolicy
import conf


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
        self.action_choice = action_choice
        self.observation_choice = observation_choice
        self.learn_total_timesteps = learn_total_timesteps
        self.clargs = clargs

    # Train and test
    def exp_training(
        self,
    ) -> None:
        """Experiment training

        @param model_filename (str): Name of the saved model.
        """

        print("#######################")
        print("START MODEL TRAINING")
        print("#######################")

        def get_env():
            env = FarmsGym(
                timestep=self.sim_options.timestep,
                observation_choice=self.observation_choice,
                action_choice=self.action_choice,
                animat_options=self.animat_options,
                arena_options=self.arena_options,
                sim_options=self.sim_options,
                simulator=self.simulator,
            )
            return env

        venv = make_vec_env(get_env, n_envs=1, seed=conf.CONF["RL"]["seed"])
        eval_venv = make_vec_env(get_env, n_envs=1, seed=conf.CONF["RL"]["seed"])
        # vec_env_cls=SubprocVecEnv

        if conf.CONF["RL"]["normWrapper"]:
            venv = VecNormalize(
                venv, norm_obs=True, norm_reward=conf.CONF["RL"]["norm_reward"]
            )
            eval_venv = VecNormalize(
                eval_venv, norm_obs=True, norm_reward=False, training=False
            )

        if "PPOparams" in conf.CONF["RL"]:
            # load optional parameters
            if "ent_coef" in conf.CONF["RL"]["PPOparams"]:
                ent_coef = conf.CONF["RL"]["PPOparams"]["ent_coef"]
            else:
                ent_coef = 0.0
            if "vf_coef" in conf.CONF["RL"]["PPOparams"]:
                vf_coef = conf.CONF["RL"]["PPOparams"]["vf_coef"]
            else:
                vf_coef = 0.5
            model = PPO(
                CustomActorCriticPolicy,
                venv,
                tensorboard_log=conf.LOG_DIR_TENSORBOARD,
                seed=conf.CONF["RL"]["seed"],
                learning_rate=linear_schedule(
                    conf.CONF["RL"]["PPOparams"]["lr_start"],
                    conf.CONF["RL"]["PPOparams"]["lr_end"],
                ),
                n_steps=conf.CONF["RL"]["PPOparams"]["n_steps"],
                batch_size=conf.CONF["RL"]["PPOparams"]["batch_size"],
                n_epochs=conf.CONF["RL"]["PPOparams"]["n_epochs"],
                gamma=conf.CONF["RL"]["PPOparams"]["gamma"],
                gae_lambda=conf.CONF["RL"]["PPOparams"]["gae_lambda"],
                use_sde=conf.CONF["RL"]["PPOparams"]["use_sde"],
                sde_sample_freq=conf.CONF["RL"]["PPOparams"]["sde_sample_freq"],
                ent_coef=ent_coef,
                vf_coef=vf_coef,
            )
        elif "SACparams" in conf.CONF["RL"]:
            model = SAC(
                "MlpPolicy",
                venv,
            )
        else:
            raise ValueError("Policy not implemented")

        # configure logger
        new_logger = configure(conf.LOG_DIR_TENSORBOARD, ["stdout", "tensorboard"])
        model.set_logger(new_logger)

        eval_callback = EvalCallback(
            eval_venv,
            eval_freq=25_000,
            deterministic=True,
            warn=True,
            verbose=1,
            # log_path=conf.LOG_DIR_TENSORBOARD, # don't know how to read the log and what's in there
            best_model_save_path=conf.LOG_DIR_RESULTS,
            callback_on_new_best=SaveVecNormalizeCallback(
                save_freq=1, name_prefix="best_model", save_path=conf.LOG_DIR_RESULTS
            )
            if conf.CONF["RL"]["normWrapper"]
            else None,
        )

        # checkpoint_callback = CheckpointCallback(
        #     save_freq=50_000,
        #     save_path=conf.LOG_DIR_RESULTS,
        #     name_prefix="checkpoint",
        #     save_replay_buffer=False,
        #     save_vecnormalize=True,
        # )

        # profile.profile(
        #     function=model.learn, total_timesteps=1, profile_filename="profile_prod_cluster_2cpu.profile"
        # )

        model.learn(
            total_timesteps=self.learn_total_timesteps,
            callback=[eval_callback],
        )
        model.save(os.path.join(conf.LOG_DIR_RESULTS, "last_model_trained.zip"))
        if conf.CONF["RL"]["normWrapper"]:
            model.get_vec_normalize_env().save(
                os.path.join(conf.LOG_DIR_RESULTS, "last_model_trained_normalize.pkl")
            )

        print("#######################")
        print("MODEL TRAINING FINISHED")
        print("#######################")

        self.test()

    def test(self) -> None:
        print("#######################")
        print("START MODEL TESTING")
        print("#######################")

        self.sim_options.record = True

        self.sim_options.n_iterations = conf.CONF[
            "n_iterations_testing"
        ]  # longer training than testing

        def get_test_env():
            test_env = FarmsGym(
                timestep=self.sim_options.timestep,
                observation_choice=self.observation_choice,
                action_choice=self.action_choice,
                animat_options=self.animat_options,
                arena_options=self.arena_options,
                sim_options=self.sim_options,
                simulator=self.simulator,
                is_test_env=True,
            )
            return test_env

        venv_test = make_vec_env(get_test_env, n_envs=1, seed=conf.CONF["RL"]["seed"])

        if conf.CONF["RL"]["normWrapper"]:
            venv_test = VecNormalize.load(
                os.path.join(conf.LOG_DIR_RESULTS, "best_model_normalize.pkl"),
                venv_test,
            )
            venv_test.training = False
            venv_test.norm_reward = False

        if "PPOparams" in conf.CONF["RL"]:
            model = PPO.load(os.path.join(conf.LOG_DIR_RESULTS, "best_model.zip"))
        elif "SACparams" in conf.CONF["RL"]:
            model = SAC.load(os.path.join(conf.LOG_DIR_RESULTS, "best_model.zip"))
        else:
            raise ValueError("Policy not implemented")

        # log gradients during one testing episode
        # onf.CONF["misc"]["log_grads"] will contain a list of all the gradients for each timestep (all outputs wrt to the inputs)
        conf.CONF["misc"]["log_grads"] = []

        rew, len_ = evaluate_policy(
            model,
            venv_test,
            n_eval_episodes=1,  # if more than one: dont log and plot gradients!
            deterministic=True,
            return_episode_rewards=True,
        )

        grads = conf.CONF["misc"]["log_grads"]  # shape (timestep, outputs, 1, inputs)
        conf.CONF["misc"]["log_grads"] = False

        rew_standardized = [10 * rew / conf.CONF['simulation_time_testing'] for rew in rew] # norm by episode length; * 10 to keep legacy experiments comparable

        # log reward of best model to performance_metrics.txt
        with open(
            os.path.join(conf.LOG_DIR_RESULTS, "performance_metrics.txt"), "a"
        ) as f:
            f.write("\n")
            f.write(f"best model reward: {rew_standardized} \n") 
        f.close()

        # log reward of best model to common results file: results.yaml
        results_file = "./experiments/results.yaml"
        results = yaml.load((open(results_file, "r")), Loader=yaml.FullLoader)
        results[conf.CONF["experiment_id"]]["best model reward"] = f"{rew_standardized}"
        with open(results_file, "w") as f:
            f.write(yaml.dump(results))
        f.close()

        # create temp dir
        _temp_dir = f"{conf.TEMP_DIR}/{1000 * time.time()}"
        os.makedirs(f"{_temp_dir}")

        # create plots of gradients
        for k in range(len(grads[0])):  # iterate over output neurons
            # TODO use numpy instead of python lists for all operations
            grads_numpy = np.array(grads, dtype=object)
            max = grads_numpy[:, k, :, :].max()
            min = grads_numpy[:, k, :, :].min()

            # save accumulated gradients
            accumulated_gradients = np.sum(np.abs(grads_numpy[:, k, :, :]), axis=(0, 1))
            fig = plt.figure(f"Accumulated gradients for output neuron {k}")
            plt.title(f"Accumulated gradients for output neuron {k}")
            plt.bar(
                [j for j in range(len(grads[0][0][0]))],
                accumulated_gradients,
                color="#072140",
            )
            plt.xticks([j for j in range(len(grads[0][0][0]))])
            plt.xlabel(f"Input neuron")
            plt.ylabel(f"Accumulated gradient")
            plt.grid(True)
            plt.savefig(
                f"{conf.LOG_DIR_RESULTS}/acc_grads_output_neuron_{k}.pdf",
            )
            plt.close()

            # create gif of gradients over episode
            for i in range(len(grads)):  # timesteps
                # generate images
                fig = plt.figure(
                    f"Gradients of output neuron {k} w.r.t. input neurons - {round(i * self.sim_options.timestep)}s"
                )
                plt.title(
                    f"Gradients of output neuron {k} w.r.t. input neurons - {round(i * self.sim_options.timestep)}s"
                )
                plt.bar(
                    [j for j in range(len(grads[i][k][0]))],
                    grads[i][k][0],
                    color="#072140",
                )
                plt.xticks([j for j in range(len(grads[i][k][0]))])
                plt.xlabel(f"Input neuron")
                plt.ylabel(f"Gradient")
                plt.grid(True)
                plt.ylim(min * 1.2, max * 1.2)
                plt.savefig(
                    f"{_temp_dir}/img_{i}.png",
                    transparent=False,
                    facecolor="white",
                )
                plt.close()

            # generate gif
            frames = []
            for t in range(self.sim_options.n_iterations):
                image = imageio.v2.imread(f"{_temp_dir}/img_{t}.png")
                frames.append(image)
            imageio.mimsave(
                f"{conf.LOG_DIR_RESULTS}/grad_anim_output_neuron_{k}.gif",
                frames,
                duration=self.sim_options.timestep * 1000,
            )

            # purge _temp_dir folder
            for f in glob.glob(f"{_temp_dir}/*"):
                os.remove(f)

        # purge and remove temp dir
        for f in glob.glob(f"{_temp_dir}/*"):
            os.remove(f)
        os.rmdir(_temp_dir)

        self.sim_options.record = False

        print("#######################")
        print("MODEL TESTING FINISHED")
        print("#######################")

        return

    # # This is another way to test a model; not used for now
    # def exp_testing(self, model_filename: str, debug_random_cond: bool) -> None:
    #     """Experiment testing

    #     @param model_filename (str): Name of the saved model.
    #     @param debug_random_cond (bool): If true, the animat is tested in random conditions.
    #     """
    #     # load trained model
    #     model = PPO.load(
    #         "experiments/051/best_model.zip",
    #     )

    #     # callback on trained model for testing
    #     gymTestCallback = GymTestCallback(
    #         timestep=self.sim_options.timestep,
    #         n_iterations=self.sim_options.n_iterations,
    #         model=model,
    #         observation_choice=self.observation_choice,
    #         action_choice=self.action_choice,
    #         debug_random_cond=False,
    #     )

    #     sim, animat_data = simulation.setup_simulation(
    #         self.animat_options,
    #         self.arena_options,
    #         self.sim_options,
    #         self.simulator,
    #         callbacks=[gymTestCallback],
    #     )

    #     gymTestCallback.set_mujoco_model(sim)

    #     print("running")

    #     sim.run()

    #     print("did run")

    #     utils.save_performance_metrics(
    #         sim,
    #         self.sim_options.timestep,
    #         self.sim_options.n_iterations,
    #     )

    #     print("saved")

    # Testing a CPG-config without a trained model, i.e. analytical
    def arch_testing(self) -> None:
        """Test the architecture of farms

        @brief: This function is used to check the options for FARMS and Notions(ExperimentOptions)
        """
        archTestCallback = ArchTestCallback()

        # self.sim_options.record = True
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
            self.sim_options.timestep,
            self.sim_options.n_iterations,
        )

        # log reward of best model to performance_metrics.txt
        with open(
            os.path.join(conf.LOG_DIR_RESULTS, "performance_metrics.txt"), "a"
        ) as f:
            f.write("\n")
            f.write(f"best model reward: {archTestCallback.reward} \n")
        f.close()

        # log reward of best model to common results file: results.yaml
        results_file = "./experiments/results.yaml"
        results = yaml.load((open(results_file, "r")), Loader=yaml.FullLoader)
        results[conf.CONF["experiment_id"]][
            "best model reward"
        ] = f"{archTestCallback.reward}"
        with open(results_file, "w") as f:
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

    def __init__(self, save_freq: int, save_path: str, name_prefix: str):
        super(SaveVecNormalizeCallback, self).__init__(0)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_normalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                print(f"#### New best model at {self.num_timesteps}")
                self.model.get_vec_normalize_env().save(path)
            else:
                raise ValueError("Error: no VecNormalize wrapper on the model")
        return True


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
