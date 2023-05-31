# This module loads the experiment configuration file
# The parameters are globally accessible by importing this module: "import conf"
import yaml
import os
from datetime import datetime

def init(experiment_config):
    global CONF
    global LOG_DIR
    global RIGHT_OSCILLATOR_INDEXES

    CONF = yaml.full_load(experiment_config)

    # log to /shared on HPC; log to ./experiments on personal PCs
    if os.path.isdir("/shared"): # cluster
        LOG_DIR = "/shared/hausdoer/experiments/" + CONF["experiment_id"] + "/logs" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        if not os.path.isdir(LOG_DIR): os.makedirs(LOG_DIR)
    else: # local
        LOG_DIR = "./experiments/" + CONF["experiment_id"] + "/logs/" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    # create log dir if not existing; not required, but just to be sure
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    RIGHT_OSCILLATOR_INDEXES = [i * 2 - 1 for i in range(1, 11)]
    LEFT_OSCILLATOR_INDEXES = [i * 2 for i in range(0, 10)]
