#!/usr/bin/env python3
"""Run a short headless FARMS/MuJoCo smoke test for a named experiment."""

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import conf
from farms_amphibious.model.options import AmphibiousArenaOptions, AmphibiousOptions
from farms_sim.simulation import setup_from_clargs
from farms_sim.utils.parse_args import sim_parse_args
from utils import simulation
from utils.limbless_experiment_options import ExperimentConditions as ExpCond
from utils.limbless_network import RobotFeedbackSenstivity


def load_experiment(path, experiment_id):
    with open(path, encoding="utf-8") as stream:
        experiments = yaml.full_load(stream)
    try:
        return experiments[experiment_id]
    except KeyError as exc:
        known = ", ".join(sorted(experiments))
        raise KeyError(f"Unknown experiment '{experiment_id}'. Known: {known}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--experiment_config",
        default="config/_EXPERIMENT/demo.yaml",
        help="YAML file containing named experiment configurations.",
    )
    parser.add_argument("-e", "--experiment_id", default="demo_arch_test")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--date", default="smoke")
    parser.add_argument("--seed", default="999")
    args = parser.parse_args()

    experiment = load_experiment(args.experiment_config, args.experiment_id)
    conf.init(experiment, args.experiment_id, None, args.date, args.seed)
    conf.CONF["log_level"] = "min"

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

    _, animat_options, sim_options, arena_options, simulator = setup_from_clargs(
        clargs=clargs,
        animat_options_loader=AmphibiousOptions,
        arena_options_loader=AmphibiousArenaOptions,
    )

    sim_options.headless = True
    sim_options.show_progress = False
    sim_options.n_iterations = args.steps
    sim_options.video = ""

    drive = conf.CONF["config"].get("drive", 2.5)
    animat_options.control.network.drives[0].initial_value = drive
    animat_options.control.network.drives[1].initial_value = drive

    exp_cond, _ = ExpCond.rlExp_sCaudal_ncCPG(
        s_caudl_senstivity=getattr(
            RobotFeedbackSenstivity, conf.CONF["robot_arch"]["s_caudl_senstivity"]
        ),
        s_local_senstivity=getattr(
            RobotFeedbackSenstivity, conf.CONF["robot_arch"]["s_local_senstivity"]
        ),
        s_caudl_weight=conf.CONF["robot_arch"]["s_caudl_weight"],
        s_local_weight=conf.CONF["robot_arch"]["s_local_weight"],
        init_osci_cond=conf.CONF["robot_arch"]["init_osci_cond"],
        c_inter=conf.CONF["robot_arch"]["c_inter"],
    )
    exp_cond.setup(animat_options)

    sim, _ = simulation.setup_simulation(
        animat_options, arena_options, sim_options, simulator, callbacks=[]
    )
    sim._env.reset()
    for step in range(args.steps):
        timestep = sim._env.step(action=None)
        if timestep.last():
            break

    print(f"Smoke test OK: stepped {step + 1} iterations for {args.experiment_id}.")


if __name__ == "__main__":
    main()
