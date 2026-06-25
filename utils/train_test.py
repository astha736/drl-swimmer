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

from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

from farms_core.utils import profile
from . import utils
from .evaluation import evaluate_policy_with_metrics

from rlgym.policies import CustomActorCriticPolicy
from utils.limbless_spawn import RobotInitialState
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

    def _get_env(self):
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

    def _get_test_env(self):
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

    def _get_eval_env(self):
        eval_env = FarmsGym(
            timestep=self.sim_options.timestep,
            observation_choice=self.observation_choice,
            action_choice=self.action_choice,
            animat_options=self.animat_options,
            arena_options=self.arena_options,
            sim_options=self.sim_options,
            simulator=self.simulator,
            is_eval_env=True,
        )
        return eval_env

    # Train and test
    def exp_training(
        self,
    ) -> None:
        """Experiment training"""

        print("#######################")
        print("START MODEL TRAINING")
        print("#######################")

        venv = make_vec_env(self._get_env, n_envs=1, seed=conf.SEED)
        # vec_env_cls=SubprocVecEnv

        if conf.CONF["RL"]["normWrapper"]:
            venv = VecNormalize(
                venv, norm_obs=True, norm_reward=conf.CONF["RL"]["norm_reward"]
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
                seed=conf.SEED,
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

        training_options = conf.CONF.get("training", {})
        eval_callback = EvalCallback(
            venv,
            eval_freq=training_options.get("eval_freq", 200_000),
            deterministic=True,
            warn=True,
            verbose=1,
            n_eval_episodes=training_options.get("eval_episodes", 20),
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
        #     function=model.learn, total_timesteps=20_000, profile_filename="profile_prod_cluster_450_20-07-2023.profile"
        # )

        callbacks = [eval_callback]
        if conf.CONF["save_observations"]:
            callbacks.append(SaveRolloutObservationsCallback())
        if conf.CONF["RL"]["curriculum"]["level"] in [2, 3, 4, 5, 6, 7]:
            callbacks.append(CurriculumStageCallback())

        model.learn(
            total_timesteps=self.learn_total_timesteps,
            callback=callbacks,
        )
        model.save(os.path.join(conf.LOG_DIR_RESULTS, "last_model_trained.zip"))
        best_model_path = os.path.join(conf.LOG_DIR_RESULTS, "best_model.zip")
        if not os.path.exists(best_model_path):
            model.save(best_model_path)
        if conf.CONF["RL"]["normWrapper"]:
            normalize_path = os.path.join(
                conf.LOG_DIR_RESULTS, "last_model_trained_normalize.pkl"
            )
            model.get_vec_normalize_env().save(normalize_path)
            best_normalize_path = os.path.join(
                conf.LOG_DIR_RESULTS, "best_model_normalize.pkl"
            )
            if not os.path.exists(best_normalize_path):
                model.get_vec_normalize_env().save(best_normalize_path)

        print("#######################")
        print("MODEL TRAINING FINISHED")
        print("#######################")

        if conf.CONF.get("post_training", {}).get("run_test", True):
            self.test()

    def test(self) -> None:
        print("#######################")
        print("START MODEL TESTING")
        print("#######################")

        self.sim_options.n_iterations = conf.CONF["n_iterations_testing"]

        conf.CONF["RL"]["curriculum"]["current_stage"] = "testing"

        self.sim_options.record = False

        eval_venv = make_vec_env(
            self._get_eval_env, n_envs=1, seed=123
        )  # fixed seed for evaluation

        if conf.CONF["RL"]["normWrapper"]:
            eval_venv = VecNormalize.load(
                os.path.join(conf.LOG_DIR_RESULTS, "best_model_normalize.pkl"),
                eval_venv,
            )
            eval_venv.training = False
            eval_venv.norm_reward = False

        if "PPOparams" in conf.CONF["RL"]:
            model = PPO.load(os.path.join(conf.LOG_DIR_RESULTS, "best_model.zip"))
        elif "SACparams" in conf.CONF["RL"]:
            model = SAC.load(os.path.join(conf.LOG_DIR_RESULTS, "best_model.zip"))
        else:
            raise ValueError("Policy not implemented")

        conf.CONF["misc"]["log_grads"] = False

        n_eval_episodes = conf.CONF.get("evaluation", {}).get("n_eval_episodes", 100)
        mean_rew, std_rew, metrics = evaluate_policy_with_metrics(
            model,
            eval_venv,
            n_eval_episodes=n_eval_episodes,  # if more than one: dont log and plot gradients!
            deterministic=True,
            return_episode_rewards=False,
            custom_metrics=True,
        )

        metrics[
            "1_standardized_reward"
        ] = f"[{10 * mean_rew / conf.CONF['simulation_time_testing']}, {10 * std_rew / conf.CONF['simulation_time_testing']}]"

        metrics["0_n_eval_episodes"] = n_eval_episodes
        metrics["0_data_format"] = "[mean, std]"
        metrics["0_seed"] = conf.SEED

        with open(
            os.path.join(f"{conf.LOG_DIR_RESULTS}", "eval_metrics.yaml"), "w"
        ) as f:
            yaml.dump(metrics, f)

        # Optionally record the single deterministic test episode.
        # ``record`` controls saving; ``video`` creates the camera callback.
        record_video = conf.CONF.get("post_training", {}).get("record_video", False)
        self.sim_options.record = record_video
        self.sim_options.video = record_video

        # reset animat_options (required because random sampling of init cond. during training)
        RobotInitialState.set_initial_conditions_parallel(
            animat_options=self.animat_options
        )

        venv_test = make_vec_env(
            self._get_test_env, n_envs=1, seed=123
        )  # fixed seed for testing

        if conf.CONF["RL"]["normWrapper"]:
            venv_test = VecNormalize.load(
                os.path.join(conf.LOG_DIR_RESULTS, "best_model_normalize.pkl"),
                venv_test,
            )
            venv_test.training = False
            venv_test.norm_reward = False

        # log gradients during one testing episode
        # conf.CONF["misc"]["log_grads"] will contain a list of all the gradients for each timestep (all outputs wrt to the inputs)
        conf.CONF["misc"]["log_grads"] = []

        rew, len_, _ = evaluate_policy_with_metrics(
            model,
            venv_test,
            n_eval_episodes=1,  # if more than one: dont log and plot gradients!
            deterministic=True,
            return_episode_rewards=True,
            log_gradients=True,
        )

        grads = conf.CONF["misc"]["log_grads"]  # shape (timestep, outputs, 1, inputs)
        conf.CONF["misc"]["log_grads"] = False

        rew_standardized = [
            10 * rew / conf.CONF["simulation_time_testing"] for rew in rew
        ]  # norm by episode length; * 10 to keep legacy experiments comparable

        # append reward of best model and #_trainable_params to single_test_env_metrics.txt
        with open(
            os.path.join(conf.LOG_DIR_RESULTS, "single_test_env_metrics.txt"), "a"
        ) as f:
            f.write("\n")
            f.write(f"best model reward standardized: {rew_standardized}")
            f.write("\n")
            f.write(
                f"policy network(s) # trainable params: {conf.CONF['misc']['log_num_trainable_params']}"
            )
        f.close()

        if conf.CONF["log_level"] == "max":
            # create temp dir
            _temp_dir = f"{conf.TEMP_DIR}/{conf.SEED}/{1000 * time.time()}"
            os.makedirs(f"{_temp_dir}")

            # create plots of gradients
            for k in range(len(grads[0])):  # iterate over output neurons
                # TODO use numpy instead of python lists for all operations
                grads_numpy = np.array(grads, dtype=object)
                max = grads_numpy[:, k, :, :].max()
                min = grads_numpy[:, k, :, :].min()

                # save accumulated gradients
                accumulated_gradients = np.sum(
                    np.abs(grads_numpy[:, k, :, :]), axis=(0, 1)
                )
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
        self.sim_options.video = False

        print("#######################")
        print("MODEL TESTING FINISHED")
        print("#######################")

        if conf.CONF.get("evaluation", {}).get("cross_seed_eval", True):
            self.cross_seed_eval()

        return

    def cross_seed_eval(self) -> None:
        print("#######################")
        print("START CROSS SEED EVAL")
        print("#######################")
        os.system(
            f"python3 utils/cross_seed_evalution.py -m={conf.EVAL_PATH_CROSS_SEED}"
        )
        print("#######################")
        print("CROSS SEED EVAL FINISHED")
        print("#######################")

    # Testing a CPG-config without a trained model, i.e. analytical
    def arch_testing(self) -> None:
        """Test the architecture of farms

        @brief: This function is used to check the options for FARMS and Notions(ExperimentOptions)
        """

        self.sim_options.n_iterations = conf.CONF["n_iterations_testing"]

        archTestCallback = ArchTestCallback()

        # self.sim_options.record = True
        sim, animat_data = simulation.setup_simulation(
            self.animat_options,
            self.arena_options,
            self.sim_options,
            self.simulator,
            callbacks=[archTestCallback],
        )

        metrics = {}
        plots = {}
        n_eval_episodes = conf.CONF.get("evaluation", {}).get("n_eval_episodes", 100)

        for i in range(n_eval_episodes):
            sim._env.reset()
            sim.run()

            _metrics, plots = utils.get_performance_metrics(
                sim,
                self.sim_options.timestep,
                self.sim_options.n_iterations,
                do_plots=(i == n_eval_episodes - 1),  # return plots on last evaluation
            )

            # standardized reward
            _metrics["reward"] = (
                10 * archTestCallback.reward / conf.CONF["simulation_time_testing"]
            )

            for metric in _metrics:
                if not metric in metrics:
                    metrics[metric] = []
                metrics[metric].append(_metrics[metric])

        for metric in metrics:
            metrics[
                metric
            ] = f"[{np.mean(metrics[metric]):.4f}, {np.std(metrics[metric]):.4f}]"

        metrics["0_n_eval_episodes"] = n_eval_episodes
        metrics["0_data_format"] = "[mean, std]"

        utils.save_performance_metrics(metrics, plots)


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
            if self.model.get_vec_normalize_env() is not None:
                print(f"#### New best model at {self.num_timesteps}")
                path = os.path.join(self.save_path, f"{self.name_prefix}_normalize.pkl")
                self.model.get_vec_normalize_env().save(path)
                # save best model for every stage in CL
                if not conf.CONF["RL"]["curriculum"]["level"] == False:
                    path = os.path.join(
                        self.save_path,
                        f"{self.name_prefix}_stage{conf.CONF['RL']['curriculum']['current_stage']}_normalize.pkl",
                    )
                    self.model.get_vec_normalize_env().save(path)
            else:
                raise ValueError("Error: no VecNormalize wrapper on the model")
        return True


class SaveRolloutObservationsCallback(BaseCallback):
    def __init__(self):
        super().__init__(0)
        self.num_rollout_buffers = 0

    def _on_rollout_end(self) -> None:
        self.num_rollout_buffers += 1
        os.makedirs(conf.LOG_DIR_OBSERVATION_BUFFER, exist_ok=True)
        np.save(
            os.path.join(
                conf.LOG_DIR_OBSERVATION_BUFFER, str(self.num_rollout_buffers)
            ),
            self.model.rollout_buffer.observations,
        )

    def _on_step(self) -> bool:
        return True


class CurriculumStageCallback(BaseCallback):
    def __init__(self):
        super().__init__(0)
        self._stage0_applied = False

    def _set_drive_trainable(self, trainable: bool) -> None:
        policy_net_drive = getattr(
            self.model.policy.mlp_extractor, "policy_net_drive", None
        )
        if policy_net_drive is None:
            raise AttributeError(
                "Curriculum training expects policy_net_drive on the custom policy."
            )
        for param in policy_net_drive.parameters():
            param.requires_grad = trainable

    def _on_rollout_start(self) -> None:
        curriculum = conf.CONF["RL"]["curriculum"]
        if curriculum["level"] == 2:
            raise NotImplementedError

        if curriculum["level"] not in [3, 4, 5, 6, 7]:
            return

        if self.num_timesteps == 0 and not self._stage0_applied:
            self._set_drive_trainable(False)
            self._stage0_applied = True
            print("Switched to CL stage #0")

        if (
            self.num_timesteps > curriculum["timesteps_stage_switch"]
            and curriculum["current_stage"] != 1
        ):
            curriculum["current_stage"] = 1
            self._set_drive_trainable(True)
            print("Switched to CL stage #1")

    def _on_step(self) -> bool:
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
