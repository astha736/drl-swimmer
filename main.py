#!/usr/bin/env python3
"""Run salamander simulation"""
import os
from re import X
import argparse

from farms_core.io.yaml import pyobject2yaml
from farms_sim.utils.parse_args import sim_parse_args
from farms_sim.simulation import setup_from_clargs
from farms_amphibious.model.options import AmphibiousOptions, AmphibiousArenaOptions
from numpy import require
from rlgym.rl_gym import ActionChoice, ObservationChoice, ObservationType, ActionType

from utils.limbless_experiment_options import ExperimentConditions as ExpCond
from utils.limbless_network import RobotFeedbackSenstivity
from utils.train_test import TrainTestClass

import conf

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment_id", required=False, default=None)
parser.add_argument("-m", "--base_test_path", required=False, default=None)
args = parser.parse_args()

# santity check on args
if args.experiment_id is None and args.base_test_path is None:
    raise ValueError("Provide either experiment_id or base_test_path.")

if args.base_test_path is not None and args.experiment_id is None:
    raise ValueError("Provide experiment_id if base_test_path is provided.")


# Load experiment config and setup *_DIRs
with open(f"./experiments/{args.experiment_id}/" + "conf.yaml") as experiment_config:
    conf.init(experiment_config, args.experiment_id, args.base_test_path)


def main() -> None:
    # setup clargs
    clargs = sim_parse_args()
    clargs.animat_config = conf.CONF["config"]["animat"]
    clargs.arena_config = conf.CONF["config"]["arena"]
    clargs.simulation_config = conf.CONF["config"]["simulation"]
    clargs.profile = os.path.join(conf.LOG_DIR_RESULTS, "simulation.profile")
    clargs.log_path = conf.LOG_DIR_RESULTS
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

    # change config params, if available
    if "drive" in conf.CONF["config"]:
        animat_options.control.network.drives[0].initial_value = conf.CONF["config"][
            "drive"
        ]
        animat_options.control.network.drives[1].initial_value = conf.CONF["config"][
            "drive"
        ]

    if "n_iterations" in conf.CONF["config"]:
        sim_options.n_iterations = conf.CONF["config"]["n_iterations"]
    else:
        conf.CONF["config"]["n_iterations"] = sim_options.n_iterations

    if "timestep" in conf.CONF["config"]:
        sim_options.timestep = conf.CONF["config"]["timestep"]
    else:
        conf.CONF["config"]["timestep"] = sim_options.timestep

    conf.CONF["config"]["simulation_time"] = (
        sim_options.timestep * sim_options.n_iterations
    )

    exp_cond_experiment, exp_cond_name = ExpCond.rlExp_sCaudal_ncCPG(
        s_caudl_senstivity=getattr(
            RobotFeedbackSenstivity,
            conf.CONF["robot_arch"]["s_caudl_senstivity"],
        ),
        s_caudl_weight=conf.CONF["robot_arch"]["s_caudl_weight"],
        init_osci_cond=conf.CONF["robot_arch"]["init_osci_cond"],
        c_inter=conf.CONF["robot_arch"]["c_inter"],
    )

    exp_cond_experiment.setup(animat_options)

    total_timesteps = (
        sim_options.n_iterations * conf.CONF["RL"]["episodes_per_training"]
    )

    # Set action and observation spaces
    action_list = []
    if "STRETCH" in conf.CONF["RL"]["action_choice"]:
        action_list.append(ActionType.STRETCH)
    if "DRIVE" in conf.CONF["RL"]["action_choice"]:
        action_list.append(ActionType.DRIVE)

    observation_list = []
    if "REACTION_XY" in conf.CONF["RL"]["observation_choice"]:
        observation_list.append(ObservationType.REACTION_XY)
    if "JOINT_POSITION" in conf.CONF["RL"]["observation_choice"]:
        observation_list.append(ObservationType.JOINT_POSITION)
    if "PHASES" in conf.CONF["RL"]["observation_choice"]:
        observation_list.append(ObservationType.PHASES)
    if "VELOCITIES" in conf.CONF["RL"]["observation_choice"]:
        observation_list.append(ObservationType.VELOCITIES)
    if "AMPLITUDES" in conf.CONF["RL"]["observation_choice"]:
        observation_list.append(ObservationType.AMPLITUDES)

    # Setup the TrainTest class
    train_test = TrainTestClass(
        animat_options=animat_options,
        arena_options=arena_options,
        sim_options=sim_options,
        simulator=simulator,
        action_choice=ActionChoice(action_list),
        observation_choice=ObservationChoice(observation_list),
        learn_total_timesteps=total_timesteps,
        clargs=clargs,
    )

    # Run experiment
    match conf.CONF["run_type"]:
        case "train":
            if args.base_test_path is not None:
                train_test.test()
            else:
                train_test.exp_training()
        case "arch_testing":
            train_test.arch_testing()
        case _:
            raise ValueError("Invalid run_type")


if __name__ == "__main__":
    main()
