# This module loads the experiment configuration file
# The parameters are globally accessible by importing this module: "import conf"
from torch import Value
import yaml
import os
from datetime import datetime


def init(experiment_config, experiment_id, base_test_path, date, seed):
    global CONF
    global LOG_DIR_RESULTS
    global LOG_DIR_TENSORBOARD
    global TEMP_DIR
    global RIGHT_OSCILLATOR_INDEXES
    global LEFT_OSCILLATOR_INDEXES
    global SEED

    RIGHT_OSCILLATOR_INDEXES = [i * 2 - 1 for i in range(1, 11)]
    LEFT_OSCILLATOR_INDEXES = [i * 2 for i in range(0, 10)]

    try:
        CONF = yaml.full_load(experiment_config)
    except:
        # we assume that experiment_config is a python object already
        CONF = experiment_config
        pass

    CONF["experiment_id"] = experiment_id
    CONF["misc"] = {}

    if not os.path.isdir(f"./experiments/{experiment_id}"):
        os.makedirs(f"./experiments/{experiment_id}")

    # if base_test_path: only tests should be run on the model within that path
    if base_test_path is not None:
        if not os.path.isdir(base_test_path):
            raise ValueError("base_test_path does not exist.")
        LOG_DIR_RESULTS = base_test_path
        SEED = int(base_test_path.rsplit("/", 1)[1])
        print(
            f"Perform only testing on models in path: {base_test_path}. Seed is {SEED}."
        )

    else:  # else: run training and testing
        # log results to home/.../experiments on local PC
        # log tensorboard logs to /shared/.. if on HPC
        SEED = int(seed)

        print(f"Perform training and testing.")
        LOG_DIR_RESULTS = (
            "./experiments/"
            + CONF["experiment_id"]
            + "/logs/"
            + str(date)
            + "/"
            + str(seed)
        )
        if os.path.isdir("/shared"):  # cluster
            LOG_DIR_TENSORBOARD = (
                "/shared/hausdoer/experiments/"
                + CONF["experiment_id"]
                + "/logs/"
                + str(date)
                + "/"
                + str(seed)
            )
            if "save_observations" in CONF:
                if CONF["save_observations"] == True:
                    global LOG_DIR_OBSERVATION_BUFFER
                    LOG_DIR_OBSERVATION_BUFFER = (
                        "/shared/hausdoer/observation_buffers/"
                        + CONF["experiment_id"]
                        + "/"
                        + str(seed)
                    )
                    if not os.path.isdir(LOG_DIR_OBSERVATION_BUFFER):
                        os.makedirs(LOG_DIR_OBSERVATION_BUFFER)
                    else:
                        import glob

                        for f in glob.glob(f"{LOG_DIR_OBSERVATION_BUFFER}/*"):
                            os.remove(f)
        else:
            LOG_DIR_TENSORBOARD = LOG_DIR_RESULTS

        if not os.path.isdir(LOG_DIR_TENSORBOARD):
            os.makedirs(LOG_DIR_TENSORBOARD)
        elif SEED == 999:
            # reserved for testing w/ debugger
            print("######## IN DEBUG MODE ##############")
        else:
            raise ValueError(
                "Log dir tb already exists. Possibly overwrite existing data. Aborted."
            )

        print(f"LOG_DIR_RESULTS: {LOG_DIR_RESULTS}")
        print(f"LOG_DIR_TENSORBOARD: {LOG_DIR_TENSORBOARD}")

    if not os.path.isdir(LOG_DIR_RESULTS):
        os.makedirs(LOG_DIR_RESULTS)

    # create _temp folder if not existent. But dont delete it if it exists, it might be used by other processes
    if os.path.isdir("/shared"):  # cluster
        TEMP_DIR = "/shared/hausdoer/_temp"
    else:
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

    # set default robot architecture (this is the default for rl)
    if not "robot_arch" in CONF:
        CONF["robot_arch"] = {}
    if not "init_osci_cond" in CONF["robot_arch"]:
        CONF["robot_arch"][
            "init_osci_cond"
        ] = (
            -1
        )  # 0 is ideal starting cond.; 1 is random starting cond.; -1 is fixed preset
    if not "s_local_weight" in CONF["robot_arch"]:
        CONF["robot_arch"]["s_local_weight"] = None  # -10 works well
    if not "s_caudl_weight" in CONF["robot_arch"]:
        CONF["robot_arch"]["s_caudl_weight"] = 0.0  # -10 works well
    if not "c_inter" in CONF["robot_arch"]:
        CONF["robot_arch"]["c_inter"] = 0  # 10 works well
    if not "s_caudl_senstivity" in CONF["robot_arch"]:
        CONF["robot_arch"][
            "s_caudl_senstivity"
        ] = "NONPERIOD"  # [SIN, COS, NONPERIOD, SIGNONE]"
    if not "s_local_senstivity" in CONF["robot_arch"]:
        CONF["robot_arch"][
            "s_local_senstivity"
        ] = "NONPERIOD"  # [SIN, COS, NONPERIOD, SIGNONE]"
    # check that exactly ONE of s_local_weight OR s_caudl_weight is not None
    if (
        CONF["robot_arch"]["s_local_weight"] == None
        and CONF["robot_arch"]["s_caudl_weight"] == None
    ):
        raise ValueError("Check robot arch 1")
    if (
        not CONF["robot_arch"]["s_local_weight"] == None
        and not CONF["robot_arch"]["s_caudl_weight"] == None
    ):
        raise ValueError("Check robot arch 2")

    # rl settings
    if "RL" in CONF:
        if not "episodes_per_training" in CONF["RL"]:
            CONF["RL"]["episodes_per_training"] = 10_000
        if not "localFeedback" in CONF["RL"]:
            CONF["RL"]["localFeedback"] = None
        # value network is equal to policy_network by default
        if not "value_network" in CONF["RL"]:
            CONF["RL"]["value_network"] = CONF["RL"]["policy_network"]
        if not "norm_reward" in CONF["RL"]:
            CONF["RL"]["norm_reward"] = True
        if not "useRandStartCondPhases" in CONF["RL"]:
            CONF["RL"]["useRandStartCondPhases"] = 2  # default
        if not "useRandStartCond" in CONF["RL"]:
            CONF["RL"]["useRandStartCond"] = 3
        if not "sample_target_velocity_from_speed_range" in CONF["RL"]:
            CONF["RL"]["sample_target_velocity_from_speed_range"] = False
        else:
            CONF["RL"]["target_velocity"] = [0.0, 0.0]
        if not "sample_init_velocity_from_speed_range" in CONF["RL"]:
            CONF["RL"]["sample_init_velocity_from_speed_range"] = False
        if not "randomInitDrive" in CONF["RL"]:
            CONF["RL"]["randomInitDrive"] = False
        if not "curriculum" in CONF["RL"]:
            CONF["RL"]["curriculum"] = {}
            CONF["RL"]["curriculum"]["level"] = False
            CONF["RL"]["curriculum"]["current_stage"] = False
        else:
            CONF["RL"]["curriculum"]["current_stage"] = 0
        if (
            CONF["RL"]["curriculum"]["level"] == 2
            or CONF["RL"]["curriculum"]["level"] == 3
            or CONF["RL"]["curriculum"]["level"] == 4
            or CONF["RL"]["curriculum"]["level"] == 5
        ):
            CONF["misc"]["CL_settings"] = {}
            for key in [
                "RewardFnc",
                "randomInitDrive",
                "sample_init_velocity_from_speed_range",
            ]:
                CONF["misc"]["CL_settings"][key] = CONF["RL"][key]

    # other settings
    if not "save_observations" in CONF:
        CONF["save_observations"] = False
    if not "stretch_action_output_scaling" in CONF:
        CONF["stretch_action_output_scaling"] = 3
    if not "frames_per_action" in CONF:
        CONF["frames_per_action"] = 1

    # sanity checks
    if (
        "stateHistoryController" in CONF["RL"]
        and not "state_history_length" in CONF["RL"]
    ):
        raise ValueError("State history controller not fully specified 1.")
    if (
        not "stateHistoryController" in CONF["RL"]
        and "state_history_length" in CONF["RL"]
    ):
        raise ValueError("State history controller not fully specified 2.")

    if not "n_iterations_testing" in CONF:
        CONF["n_iterations_testing"] = 2500
    if not "n_iterations" in CONF:
        CONF["n_iterations"] = 1000

    CONF["testing_transient"] = 3.0  # s

    CONF["misc"]["log_grads"] = False
    CONF["misc"]["log_num_trainable_params"] = False
