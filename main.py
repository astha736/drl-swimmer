#!/usr/bin/env python3
"""Run salamander simulation"""
import os
from re import X
from datetime import datetime
import yaml
import argparse

from farms_core.io.yaml import pyobject2yaml
from farms_sim.utils.parse_args import sim_parse_args
from farms_sim.simulation import setup_from_clargs
from farms_amphibious.model.options import AmphibiousOptions, AmphibiousArenaOptions
from rlgym.rl_gym import ActionChoice, ObservationChoice, ObservationType, ActionType

from utils.limbless_experiment_options import ExperimentConditions as ExpCond
from utils.limbless_network import RobotFeedbackSenstivity
from utils.train_test import TrainTestClass


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("experiment_id")
args = parser.parse_args()


# Load experiment config and setup *_DIRs
with open(f"./experiments/{args.experiment_id}/" + "conf.yaml") as experiment_config:
    conf = yaml.full_load(experiment_config)
LOG_DIR = (
    "./experiments/"
    + conf["experiment_id"]
    + "/logs/"
    # + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
)


def main() -> None:

    # setup clargs
    clargs = sim_parse_args()
    clargs.animat_config = conf["config"]["animat"]
    clargs.arena_config = conf["config"]["arena"]
    clargs.simulation_config = conf["config"]["simulation"]
    clargs.profile = os.path.join(LOG_DIR, "simulation.profile")
    clargs.log_path = LOG_DIR
    clargs.prompt = False
    clargs.simulator = "MUJOCO"
    clargs.test_configs = False
    clargs.verify_save = False

    # create log dir if not existing
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

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
    match conf["robot"]["base_config"]:
        case "default":
            exp_cond_experiment, exp_cond_name = ExpCond.rlExp_sCaudal_ncCPG()
        case "arch_testing":
            exp_cond_experiment, exp_cond_name = ExpCond.rlExp_sCaudal_ncCPG(
                s_caudl_senstivity=getattr(
                    RobotFeedbackSenstivity,
                    conf["robot"]["s_caudl_senstivity"],
                ),
                s_caudl_weight=conf["robot"]["s_caudl_weight"],
                init_osci_cond=conf["robot"]["init_osci_cond"],
                c_inter=conf["robot"]["c_inter"],
            )
        case _:
            raise ValueError("Invalid robot_configuration")
    exp_cond_experiment.setup(animat_options)

    total_timesteps = sim_options.n_iterations * conf["RL"]["episodes_per_training"]

    # Set action and observation spaces
    action_list = []
    if "STRETCH" in conf["RL"]["action_choice"]:
        action_list.append(ActionType.STRETCH)

    observation_list = []
    if "REACTION_XY" in conf["RL"]["observation_choice"]:
        observation_list.append(ObservationType.REACTION_XY)
    if "JOINT_POSITION" in conf["RL"]["observation_choice"]:
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
        experiment_args=conf,
        experiment_id=args.experiment_id,
        clargs=clargs,
    )

    # Run experiment
    match conf["run_type"]:
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
