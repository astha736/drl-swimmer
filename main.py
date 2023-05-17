#!/usr/bin/env python3
"""Run salamander simulation"""
import os
from re import X
from datetime import datetime
import yaml

from farms_core.io.yaml import pyobject2yaml
from farms_sim.utils.parse_args import sim_parse_args
from farms_sim.simulation import setup_from_clargs
from farms_amphibious.model.options import AmphibiousOptions, AmphibiousArenaOptions
from rlgym.rl_gym import ActionChoice, ObservationChoice, ObservationType, ActionType

from utils.limbless_experiment_options import ExperimentConditions as ExpCond
from utils.limbless_network import RobotFeedbackSenstivity
from utils.train_test import TrainTestClass

# Load experiment config and setup *_DIRs
with open("./experiments/experiment_01/" + "experiment_01.yaml") as exp_config:
    args = yaml.full_load(exp_config)
LOGS_DIR = (
    "./experiments/experiment_"
    + args["experiment_id"]
    + "/logs/"
    + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
)


def main() -> None:

    # setup clargs
    clargs = sim_parse_args()
    clargs.animat_config = args["configs"]["animat_config"]
    clargs.arena_config = args["configs"]["arena_config"]
    clargs.simulation_config = args["configs"]["simulation_config"]
    clargs.profile = os.path.join(LOGS_DIR, "simulation.profile")
    clargs.log_path = LOGS_DIR
    clargs.prompt = False
    clargs.simulator = "MUJOCO"
    clargs.test_configs = False
    clargs.verify_save = False

    # Load options from yaml files
    (
        clargs,
        animat_options,
        sim_options,
        arena_options,
        simulator,
    ) = setup_from_clargs(
        clargs=clargs,
        animat_options_loader=AmphibiousOptions,
        arena_options_loader=AmphibiousArenaOptions,
    )

    # Load experiment conditions
    match args["robot_configuration"]:
        case "default":
            exp_cond_experiment, exp_cond_name = ExpCond.rlExp_sCaudal_ncCPG()
        case "arch_testing":
            exp_cond_experiment, exp_cond_name = ExpCond.rlExp_sCaudal_ncCPG(
                s_caudl_senstivity=RobotFeedbackSenstivity.SIN, s_caudl_weight=-10
            )
        case _:
            raise ValueError("Invalid robot_configuration")
    exp_cond_experiment.setup(animat_options)

    # Set simulation options
    sim_options.fast = True
    sim_options.headless = False
    sim_options.n_iterations = args["simulation"][
        "n_iterations"
    ]  # timesteps per episode
    sim_options.timestep = args["simulation"]["timestep"]
    total_timesteps = sim_options.n_iterations * args["RL"]["episodes_per_training"]

    # Set action and observation spaces
    action_list = []
    if "STRETCH" in args["RL"]["action_choice"]:
        action_list.append(ActionType.STRETCH)

    observation_list = []
    if "REACTION_XY" in args["RL"]["observation_choice"]:
        observation_list.append(ObservationType.REACTION_XY)
    if "JOINT_POSITION" in args["RL"]["observation_choice"]:
        observation_list.append(ObservationType.JOINT_POSITION)

    # Setup the TrainTest class
    train_test = TrainTestClass(
        animat_options=animat_options,
        arena_options=arena_options,
        sim_options=sim_options,
        simulator=simulator,
        log_dir=clargs.log_path,
        action_choice=ActionChoice(action_list),
        observation_choice=ObservationChoice(observation_list),
        learn_total_timesteps=total_timesteps,
        experiment_args=args,
        clargs=clargs,
    )

    # Run experiment
    match args["run_type"]:
        case "train":
            train_test.exp_training(model_filename=exp_cond_name)
        case "test":
            train_test.exp_testing(
                model_filename=exp_cond_name, debug_random_cond=False
            )
        case "arch_testing":
            train_test.arch_testing()
        case _:
            raise ValueError("Invalid run_type")


if __name__ == "__main__":
    main()
