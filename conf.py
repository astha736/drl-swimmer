# This module loads the experiment configuration file
# The parameters are globally accessible by importing this module: "import conf"
from torch import Value
import yaml
import os
from datetime import datetime


def init(experiment_config, experiment_id, base_test_path):
    global CONF
    global LOG_DIR_RESULTS
    global LOG_DIR_TENSORBOARD
    global TEMP_DIR
    global RIGHT_OSCILLATOR_INDEXES
    global LEFT_OSCILLATOR_INDEXES

    RIGHT_OSCILLATOR_INDEXES = [i * 2 - 1 for i in range(1, 11)]
    LEFT_OSCILLATOR_INDEXES = [i * 2 for i in range(0, 10)]

    CONF = yaml.full_load(experiment_config)
    CONF["experiment_id"] = experiment_id

    # set log paths
    if base_test_path is not None:
        if not os.path.isdir(base_test_path):
            raise ValueError("base_test_path does not exist.")
        print(f"Perform only testing on models in path: {base_test_path}.")
        LOG_DIR_RESULTS = base_test_path
    else:
        # log results to home/.../experiments on local PC
        # log tensorboard logs to /shared/.. if on HPC
        print(f"Perform training and testing.")
        LOG_DIR_RESULTS = (
            "./experiments/"
            + CONF["experiment_id"]
            + "/logs/"
            + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        )
        if os.path.isdir("/shared"):  # cluster
            LOG_DIR_TENSORBOARD = (
                "/shared/hausdoer/experiments/"
                + CONF["experiment_id"]
                + "/logs"
                + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            )
        else:
            LOG_DIR_TENSORBOARD = LOG_DIR_RESULTS
        if not os.path.isdir(LOG_DIR_TENSORBOARD):
            os.makedirs(LOG_DIR_TENSORBOARD)

        print(f"LOG_DIR_RESULTS: {LOG_DIR_RESULTS}")
        print(f"LOG_DIR_TENSORBOARD: {LOG_DIR_TENSORBOARD}")

    if not os.path.isdir(LOG_DIR_RESULTS):
        os.makedirs(LOG_DIR_RESULTS)

    # create _temp folder or purge it
    TEMP_DIR = "./_temp"
    if not os.path.isdir(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # load referenced parameters in experiment config file
    if "PPOparams" in CONF["RL"]:
        with open(CONF["RL"]["PPOparams"]) as f:
            CONF["RL"]["PPOparams"] = yaml.full_load(f)

    if "SACparams" in CONF["RL"]:
        with open(CONF["RL"]["SACparams"]) as f:
            CONF["RL"]["SACparams"] = yaml.full_load(f)

    if "RewardFnc" in CONF["RL"]:
        with open(CONF["RL"]["RewardFnc"]) as f:
            CONF["RL"]["RewardFnc"] = yaml.full_load(f)

    # set default parameters
    if not "localFeedback" in CONF["RL"]:
        CONF["RL"]["localFeedback"] = None

    if not "robot_arch" in CONF:
        CONF["robot_arch"] = {}
        CONF["robot_arch"][
            "init_osci_cond"
        ] = (
            -1
        )  # 0 is ideal starting cond.; 1 is random starting cond.; -1 is fixed preset
        CONF["robot_arch"]["s_caudl_weight"] = 0  # -10 works well
        CONF["robot_arch"]["c_inter"] = 0  # 10 works well
        CONF["robot_arch"][
            "s_caudl_senstivity"
        ] = "NONPERIOD"  # [SIN, COS, NONPERIOD, SIGNONE]"

    if "RL" in CONF:
        # value network is equal to policy_network by default
        if not "value_network" in CONF["RL"]:
            CONF["RL"]["value_network"] = CONF["RL"]["policy_network"]

    if "RL" in CONF:
        if not "seed" in CONF["RL"]:
            CONF["RL"]["seed"] = 123

    CONF["n_iterations_testing"] = 1000

    CONF["misc"] = {}
    CONF["misc"]["log_grads"] = False
