import numpy as np

from typing import Union, Any, Dict, List, Tuple
from farms_core import pylog
from utils.limbless_network import RobotInitialNetwork, RobotFeedbackSenstivity
from utils.limbless_domain import RobotInitialDomain
from utils.limbless_spawn import RobotInitialState, RobotInitialShape
from utils.limbless_oscillator import RobotInitialOscillator
from farms_core.model.options import AnimatOptions


class RobotInitialConditions:
    """RobotInitialCondition helps setting initial conditions for experiments

    The Initial conditions are agnostic to CPG control or CPG-RL control

    Note: These are just initial conditions, and some of the initial conditions
    for example weights can be changed on the fly (for RL) by changing their
    data array. This is usually done by RL. Check the action step of RL.

    """

    def __init__(
        self,
        network_exp: RobotInitialNetwork,
        domain_exp: RobotInitialDomain,
        initial_exp: RobotInitialState,
        osc_exp: RobotInitialOscillator = None,
    ):
        self.network_exp = network_exp  # of class RobotInitialNetwork
        self.domain_exp = domain_exp  # of class RobotInitialDomain
        self.initial_exp = initial_exp  # of class RobotInitialState
        self.osc_exp = (
            RobotInitialOscillator(cond=0) if osc_exp is None else osc_exp
        )  # of class RobotInitialOscillator
        self.sim_n = None  # keeping track of the simulations with a number

    def setup(self, animat_options: AnimatOptions):
        # call corresponding setup functions
        self.network_exp.setup(animat_options)
        self.domain_exp.setup(animat_options)
        self.initial_exp.setup(animat_options)
        self.osc_exp.setup(animat_options)


class ExperimentConditions:
    @staticmethod
    def friction_OpenCPG(
        f_start: float = 0.2,
        f_end: float = 0.5,
        f_num: float = 4,
        robot_shape: RobotInitialShape = RobotInitialShape.PARALLEL,
        robot_pose: int = 0,
        **kwargs,
    ) -> Tuple[List[RobotInitialConditions], str]:
        """Design of experiment for friction"""
        experiments = []
        experiments += [
            RobotInitialConditions(
                network_exp=RobotInitialNetwork(
                    c_intra=kwargs.pop("c_intra", 10),
                    c_inter=kwargs.pop("c_inter", 10),
                    s_local=kwargs.pop("s_local", None),
                    s_rostl=kwargs.pop("s_rostl", None),
                    s_caudl=kwargs.pop("s_caudl", None),
                ),
                domain_exp=RobotInitialDomain(friction=f),
                initial_exp=RobotInitialState(
                    condition_shape=robot_shape, pose=robot_pose
                ),
            )
            for f in np.linspace(f_start, f_end, num=f_num)
        ]

        exp_name = "friction_OpenCPG"
        return experiments, exp_name

    @staticmethod
    def spawn_OpenCPG(
        robot_shapes: list = RobotInitialState.get_init_robot_state_list,
        robot_poses: list = RobotInitialState.get_init_robot_pose_list,
        friction: float = 0.2,
        **kwargs,
    ) -> Tuple[List[RobotInitialConditions], str]:
        """Design of Spawn condition for the robot"""
        experiments = []
        experiments += [
            RobotInitialConditions(
                network_exp=RobotInitialNetwork(
                    c_intra=kwargs.pop("c_intra", 10),
                    c_inter=kwargs.pop("c_inter", 10),
                    s_local=kwargs.pop("s_local", None),
                    s_rostl=kwargs.pop("s_rostl", None),
                    s_caudl=kwargs.pop("s_caudl", None),
                ),
                domain_exp=RobotInitialDomain(friction=friction),
                initial_exp=RobotInitialState(
                    condition_shape=robot_shape, pose=robot_pose
                ),
            )
            for robot_shape in robot_shapes
            for robot_pose in robot_poses
        ]
        exp_name = "spawn_OpenCPG"
        return experiments, exp_name

    @staticmethod
    def senstivity_sCaudal_ncCPG(
        s_caudal_weights_list: list = [-30, 30],
        s_caudl_senstivity_list=[
            RobotFeedbackSenstivity.COS,
            RobotFeedbackSenstivity.SIN,
            RobotFeedbackSenstivity.SIGNONE,
            RobotFeedbackSenstivity.NONPERIOD,
        ],
        robot_shape: RobotInitialShape = RobotInitialShape.PARALLEL,
        robot_pose: int = 0,
        friction: float = 0.2,
        **kwargs,
    ) -> Tuple[List[RobotInitialConditions], str]:
        """Stretch Caudal with No Coupling CPG
        Use: function to play with different senstivity functions for feedback.
        """
        experiments = []
        experiments += [
            RobotInitialConditions(
                network_exp=RobotInitialNetwork(
                    c_intra=kwargs.pop("c_intra", 10),
                    c_inter=kwargs.pop("c_inter", 0),
                    s_local=None,
                    s_rostl=None,
                    s_caudl=s_caudal_w,
                    s_caudl_senstivity=s_caudl_senstivity,
                ),
                domain_exp=RobotInitialDomain(friction=friction),
                initial_exp=RobotInitialState(
                    condition_shape=robot_shape, pose=robot_pose
                ),
            )
            for s_caudal_w in s_caudal_weights_list
            for s_caudl_senstivity in s_caudl_senstivity_list
        ]

        exp_name = "senstivity_sCaudal_ncCPG"
        return experiments, exp_name

    @staticmethod
    def tegotae_sCaudal_ncCPG(
        s_caudal_weights_list: list = [-30],
        robot_shapes: list = None,
        robot_poses: list = None,
        friction: float = 0.2,
        **kwargs,
    ) -> Tuple[List[RobotInitialConditions], str]:
        """Tegotae experiment for different weight

        Use: function to play with tegotae(sin), paramaters - weight, shapes and poses
        """

        robot_shapes = (
            [RobotInitialShape.PARALLEL] if robot_shapes is None else robot_shapes
        )
        robot_poses = [0] if robot_poses is None else robot_poses

        debug = kwargs.pop("debug", False)

        experiments = []
        experiments += [
            RobotInitialConditions(
                network_exp=RobotInitialNetwork(
                    c_intra=kwargs.pop("c_intra", 10),
                    c_inter=kwargs.pop("c_inter", 0),
                    s_local=None,
                    s_rostl=None,
                    s_caudl=s_caudal_w,
                    s_caudl_senstivity=RobotFeedbackSenstivity.SIN,
                ),
                domain_exp=RobotInitialDomain(friction=friction),
                initial_exp=RobotInitialState(
                    condition_shape=robot_shape, pose=robot_pose
                ),
            )
            for s_caudal_w in s_caudal_weights_list
            for robot_shape in robot_shapes
            for robot_pose in robot_poses
        ]

        return experiments, "tegotae_sCaudal_ncCPG"

    # Used for rl experiments
    @staticmethod
    def rlExp_sCaudal_ncCPG(
        s_caudl_senstivity: RobotFeedbackSenstivity = RobotFeedbackSenstivity.NONPERIOD,
        s_caudl_weight: float = None,  # default value for Network initialization
        s_local_senstivity: RobotFeedbackSenstivity = RobotFeedbackSenstivity.NONPERIOD,
        s_local_weight: float = None,  # default value for Network initialization
        robot_shape: RobotInitialShape = RobotInitialShape.PARALLEL,
        robot_pose: int = 0,
        friction: float = 0.2,
        debug: bool = False,
        init_osci_cond: int = -1,
        c_inter: float = 0,
        **kwargs,
    ) -> Tuple[RobotInitialConditions, str]:
        """Design for gym learning

        Use: explore stretch(caudal) + decoupled cpg

        return: experiment(one instance), exp_name

        """
        c_intra = kwargs.pop("c_intra", 10)

        experiment = RobotInitialConditions(
            network_exp=RobotInitialNetwork(
                c_intra=c_intra,
                c_inter=c_inter,
                s_local=s_local_weight,  # used for rl to address all couplings to oscillators
                s_rostl=None,
                s_caudl=s_caudl_weight,  # set to zero + ideal starting conditions for only swimming
                s_caudl_senstivity=s_caudl_senstivity,
                s_local_senstivity=s_local_senstivity,
            ),
            domain_exp=RobotInitialDomain(friction=friction),
            initial_exp=RobotInitialState(condition_shape=robot_shape, pose=robot_pose),
            osc_exp=RobotInitialOscillator(cond=init_osci_cond),
        )

        if debug:
            print("s_caudl_senstivity: {}".format(s_caudl_senstivity))
            print("robot_shape       : {}".format(robot_shape))

        exp_name = "undefined"

        return experiment, exp_name

    @staticmethod
    def rlExp_sCaudal_CPG(
        s_caudl_senstivity: RobotFeedbackSenstivity = RobotFeedbackSenstivity.NONPERIOD,
        s_caudl_weight: float = 0,  # default value for Network initialization
        robot_shape: RobotInitialShape = RobotInitialShape.PARALLEL,
        robot_pose: int = 0,
        friction: float = 0.2,
        debug: bool = False,
        **kwargs,
    ) -> Tuple[RobotInitialConditions, str]:
        """Design for gym learning

        Use: explore stretch(caudal) + cpg

        return: experiment(one instance), exp_name

        """
        c_intra = kwargs.pop("c_intra", 10)
        c_inter = kwargs.pop("c_inter", 10)

        experiment = RobotInitialConditions(
            network_exp=RobotInitialNetwork(
                c_intra=c_intra,
                c_inter=c_inter,
                s_local=None,
                s_rostl=None,
                s_caudl=s_caudl_weight,
                s_caudl_senstivity=s_caudl_senstivity,
            ),
            domain_exp=RobotInitialDomain(friction=friction),
            initial_exp=RobotInitialState(condition_shape=robot_shape, pose=robot_pose),
            osc_exp=RobotInitialOscillator(cond=0),  # ideal
        )

        if debug:
            print("s_caudl_senstivity: {}".format(s_caudl_senstivity))
            print("robot_shape       : {}".format(robot_shape))
            print("c_inter: {}".format(c_inter))
            print("c_intra: {}".format(c_intra))

        exp_name = (
            "s{}_".format(s_caudl_senstivity)
            + "sW{}_".format(int(100 * s_caudl_weight))
            + "caW{}_".format(c_intra)
            + "crW{}_".format(c_inter)
            + "sCaudal_CPG"
        )

        return experiment, exp_name

    @staticmethod
    def rlExp_sCaudal_rLocal_CPG(
        s_caudl_senstivity: RobotFeedbackSenstivity = RobotFeedbackSenstivity.NONPERIOD,
        s_caudl_weight: float = 0,  # default value for Network initialization
        robot_shape: RobotInitialShape = RobotInitialShape.PARALLEL,
        robot_pose: int = 0,
        friction: float = 0.2,
        debug: bool = False,
        **kwargs,
    ) -> Tuple[RobotInitialConditions, str]:
        """Design for gym learning

        Use: explore stretch (caudal) + reaction (local)
        reaction_local_Senstivity = REACTION2FEQ

        TODO: before using rLocal check if X/Y local is used or Z reaction is used

        """
        c_intra = kwargs.pop("c_intra", 10)
        c_inter = kwargs.pop("c_inter", 10)
        reaction_local = kwargs.pop("reaction_local", 0.0)

        experiments = RobotInitialConditions(
            network_exp=RobotInitialNetwork(
                c_intra=c_intra,
                c_inter=c_inter,
                s_local=None,
                s_rostl=None,
                s_caudl=s_caudl_weight,
                s_caudl_senstivity=s_caudl_senstivity,
                reaction_local=reaction_local,
            ),
            domain_exp=RobotInitialDomain(friction=friction),
            initial_exp=RobotInitialState(condition_shape=robot_shape, pose=robot_pose),
            osc_exp=RobotInitialOscillator(cond=0),  # ideal
        )
        if debug:
            print("s_caudl_senstivity: {}".format(s_caudl_senstivity))
            print("robot_shape       : {}".format(robot_shape))
            print("c_inter: {}".format(c_inter))
            print("c_intra: {}".format(c_intra))
            print("reaction_local: {}".format(reaction_local))

        exp_name = (
            "s{}_".format(s_caudl_senstivity)
            + "sW{}_".format(int(100 * s_caudl_weight))
            + "caW{}_".format(c_intra)
            + "crW{}_".format(c_inter)
            + "reW{}_".format(reaction_local)
            + "sCaudal_rLocal_CPG"
        )

        return experiments, exp_name


if __name__ == "__main__":
    raise ValueError("Not a file that is supposed to be run")
