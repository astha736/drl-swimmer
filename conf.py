# This module loads the experiment configuration file
# The parameters are globally accessible by importing this module: "import conf"
import yaml
import os
from datetime import datetime

def init(experiment_config, experiment_id):
    global CONF
    global LOG_DIR_RESULTS
    global LOG_DIR_TENSORBOARD
    global RIGHT_OSCILLATOR_INDEXES
    global LEFT_OSCILLATOR_INDEXES

    RIGHT_OSCILLATOR_INDEXES = [i * 2 - 1 for i in range(1, 11)]
    LEFT_OSCILLATOR_INDEXES = [i * 2 for i in range(0, 10)]

    CONF = yaml.full_load(experiment_config)
    CONF["experiment_id"] = experiment_id

    # log results to home/.../experiments on local PC
    # log tensorboard logs to /shared/.. if on HPC
    LOG_DIR_RESULTS= "./experiments/" + CONF["experiment_id"] + "/logs/" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    if os.path.isdir("/shared"): # cluster
        LOG_DIR_TENSORBOARD = "/shared/hausdoer/experiments/" + CONF["experiment_id"] + "/logs" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    else:
        LOG_DIR_TENSORBOARD = LOG_DIR_RESULTS
    if not os.path.isdir(LOG_DIR_TENSORBOARD): os.makedirs(LOG_DIR_TENSORBOARD)
    if not os.path.isdir(LOG_DIR_RESULTS): os.makedirs(LOG_DIR_RESULTS)

    # load referenced parameters in experiment config file
    if "PPOparams" in CONF["RL"]:
        with open(CONF["RL"]["PPOparams"]) as f:
            CONF["RL"]["PPOparams"] = yaml.full_load(f)
            
    if "RewardFnc" in CONF["RL"]:
        with open(CONF["RL"]["RewardFnc"]) as f:
            CONF["RL"]["RewardFnc"] = yaml.full_load(f)

   
