from enum import Enum
import numpy as np
import random


class RobotInitialShape(Enum):
    S_SHAPE = 0
    L_SHAPE = 1
    U_SHAPE = 2
    PARALLEL = 3
    DIAGONAL = 4
    TEST = 5
    RANDOM = 6


class RobotInitialState:
    """Get the options for Initialization"""

    # static attribute of the class
    _robot_state_list = [
        RobotInitialShape.S_SHAPE,
        RobotInitialShape.L_SHAPE,
        RobotInitialShape.U_SHAPE,
        RobotInitialShape.PARALLEL,
        RobotInitialShape.DIAGONAL,
    ]
    _robot_pose_list = [0, 1, 2]

    def __init__(self, condition_shape: RobotInitialShape, pose: int):
        self.condition_shape = condition_shape
        self.pose = pose

    def get_init_robot_state_list():
        return RobotInitialState._robot_state_list

    def get_init_robot_pose_list():
        return RobotInitialState._robot_pose_list

    def set_initial_conditions_Lshape(animat_options, pose=0):
        """Initial condition with L shape
        Shape of the robot is defined with the for loop
        Pose of the robot is defined as [x, y, z, r_x, r_y, r_z]

        Note: Aim of these initial  condition is to avoid collision with pegs
        when launched
        """

        # shape of the robot
        for i, elem in enumerate(animat_options.morphology.joints):
            if i > 5:
                # tail i = 7, 8, 9 (tail)
                joint_initial_pos = 0
            elif i > 3:
                # trunk i = 4, 5, 6 (trunk)
                joint_initial_pos = 0.0  # -(np.pi/7)
            else:
                # head i = 0, 1, 2, 3 (head)
                joint_initial_pos = -(np.pi / 8)
            elem.initial = [joint_initial_pos, 0]

        # pose of the robot
        if pose == 0:
            animat_options.spawn.pose = [-0.065, -0.05, 0.1, 0.0, 0.0, 0.0]
        elif pose == 1:
            animat_options.spawn.pose = [-0.060, -0.05, 0.1, 0.0, 0.0, 0.1]
        elif pose == 2:
            animat_options.spawn.pose = [-0.000, -0.05, 0.1, 0.0, 0.0, 0.2]
        elif pose == -1:
            animat_options.spawn.pose = [-2.0, -2.0, 0.1, 0.0, 0.0, 0.0]
        else:
            raise ValueError("Case for desired pose option not specified")
        return

    def set_initial_conditions_Sshape(animat_options, pose=0):
        """Initial condition with S shape
        Shape of the robot is defined with the for loop
        Pose of the robot is defined as [x, y, z, r_x, r_y, r_z]

        Note: Aim of these initial  condition is to avoid collision with pegs
        when launched
        """
        # robot shape
        for i, elem in enumerate(animat_options.morphology.joints):
            if i > 5:
                # tail i = 6, 7, 8, 9 (tail)
                joint_initial_pos = np.pi / 8
            elif i > 3:
                # trunk i = 4, 5 (trunk)
                joint_initial_pos = 0.0  # -(np.pi/7)
            else:
                # head i = 0, 1, 2, 3 (head)
                joint_initial_pos = -(np.pi / 8)
            elem.initial = [joint_initial_pos, 0]

        # robot pose
        if pose == 0:
            animat_options.spawn.pose = [-0.08, -0.00, 0.1, 0.0, 0.0, 0.0]
        elif pose == 1:
            animat_options.spawn.pose = [-0.000, -0.06, 0.1, 0.0, 0.0, 0.3]
        elif pose == 2:
            animat_options.spawn.pose = [-0.000, -0.06, 0.1, 0.0, 0.0, 0.2]
        else:
            raise ValueError("Case for desired pose option not specified")
        return

    def set_initial_conditions_Ushape(animat_options, pose=0):
        """Initial condition with U shape
        Shape of the robot is defined with the for loop
        Pose of the robot is defined as [x, y, z, r_x, r_y, r_z]

        Note: Aim of these initial  condition is to avoid collision with pegs
        when launched
        """
        # robot shape
        for i, elem in enumerate(animat_options.morphology.joints):
            if i > 4:
                # tail i = 5, 6, 7, 8, 9 (tail)
                joint_initial_pos = -(np.pi / 8)
            else:
                # head i = 0, 1, 2, 3, 4 (head)
                joint_initial_pos = -(np.pi / 8)
            elem.initial = [joint_initial_pos, 0]

        # robot pose
        if pose == 0:
            animat_options.spawn.pose = [0.07, 0.00, 0.1, 0.0, 0.0, -0.0]
        elif pose == 1:
            animat_options.spawn.pose = [0.07, 0.00, 0.1, 0.0, 0.0, -0.1]
        elif pose == 2:
            animat_options.spawn.pose = [0.07, 0.00, 0.1, 0.0, 0.0, -0.2]
        else:
            raise ValueError("Case for desired pose option not specified")
        return

    def set_initial_conditions_parallel(animat_options, pose=0):
        """Initial condition with Line shape, but placed parallel to the pegs
        Shape of the robot is defined with the for loop
        Pose of the robot is defined as [x, y, z, r_x, r_y, r_z]

        Note: Aim of these initial  condition is to avoid collision with pegs
        when launched
        """
        # robot shape
        for i, elem in enumerate(animat_options.morphology.joints):
            elem.initial = [0.0, 0]

        # robot pose
        if pose == 0:
            animat_options.spawn.pose = [0.00, -0.03, 0.1, 0.0, 0.0, -0.1]
        elif pose == 1:
            animat_options.spawn.pose = [0.00, 0.00, 0.1, 0.0, 0.0, -0.0]
        elif pose == 2:
            animat_options.spawn.pose = [-0.03, -0.03, 0.1, 0.0, 0.0, -0.1]
        elif pose == -1:
            animat_options.spawn.pose = [-2.0, -2.0, 0.1, 0.0, 0.0, 0.0]
        else:
            raise ValueError("Case for desired pose option not specified")
        return

    def set_initial_conditions_diagonal(animat_options, pose=0):
        """Initial condition with Line shape, but placed diagonal to the pegs
        Shape of the robot is defined with the for loop
        Pose of the robot is defined as [x, y, z, r_x, r_y, r_z]

        Note: Aim of these initial  condition is to avoid collision with pegs
        when launched
        """
        # robot shape
        for i, elem in enumerate(animat_options.morphology.joints):
            elem.initial = [0.0, 0]

        # robot pose
        if pose == 0:
            animat_options.spawn.pose = [0.012, 0.05, 0.1, 0.0, 0.0, -0.8]
        elif pose == 1:
            animat_options.spawn.pose = [0.07, 0.00, 0.1, 0.0, 0.0, -0.8]
        elif pose == 2:
            animat_options.spawn.pose = [0.07, 0.00, 0.1, 0.0, 0.0, -0.75]
        else:
            raise ValueError("Case for desired pose option not specified")
        return

    def set_initial_conditions_test(animat_options, pose=0):
        """Initial condition with Line shape, but placed diagonal to the pegs
        Shape of the robot is defined with the for loop
        Pose of the robot is defined as [x, y, z, r_x, r_y, r_z]

        Note: Aim of these initial  condition is to avoid collision with pegs
        when launched
        """
        # robot shape
        for i, elem in enumerate(animat_options.morphology.joints):
            elem.initial = [0.0, 0]

        # robot pose
        if pose == 0:
            animat_options.spawn.pose = [-1.5, 0.05, 0.1, 0.0, 0.0, -0.8]
        elif pose == 1:
            animat_options.spawn.pose = [-1.07, 0.00, 0.1, 0.0, 0.0, -0.8]
        elif pose == 2:
            animat_options.spawn.pose = [0.07, 0.00, 0.1, 0.0, 0.0, -0.75]
        else:
            raise ValueError("Case for desired pose option not specified")
        return

    def set_shape_and_pose(self, animat_options):
        """Class method

        Args:
            animat_options (_type_): _description_

        Raises:
            ValueError: _description_
        """
        RobotInitialState.set_shape_and_pose_static(
            animat_options=animat_options, shape=self.condition_shape, pose=self.pose
        )
        return

    @staticmethod
    def set_shape_and_pose_static(shape: RobotInitialShape, pose: int, animat_options):
        """Change the shape and pose of the robot in animat options

        TODO: typing animat_options class

        Args:
            shape (RobotInitialShape): _description_
            pose (int): _description_
            animat_options (_type_): _description_

        Raises:
            ValueError: _description_
        """
        if shape == RobotInitialShape.L_SHAPE:
            RobotInitialState.set_initial_conditions_Lshape(animat_options, pose)
        elif shape == RobotInitialShape.S_SHAPE:
            RobotInitialState.set_initial_conditions_Sshape(animat_options, pose)
        elif shape == RobotInitialShape.U_SHAPE:
            RobotInitialState.set_initial_conditions_Ushape(animat_options, pose)
        elif shape == RobotInitialShape.PARALLEL:
            RobotInitialState.set_initial_conditions_parallel(animat_options, pose)
        elif shape == RobotInitialShape.DIAGONAL:
            RobotInitialState.set_initial_conditions_diagonal(animat_options, pose)
        elif shape == RobotInitialShape.TEST:
            RobotInitialState.set_initial_conditions_test(animat_options, pose)
        elif shape == RobotInitialShape.RANDOM:
            RobotInitialState.set_random_shape_pose(animat_options)
        else:
            raise ValueError(
                " condition_shape value {}, not found. Please check again".format(shape)
            )
        return

    def set_user_defined_shape_pose(animat_options, shape_pose):
        for i, elem in enumerate(animat_options.morphology.joints):
            elem.initial = [shape_pose[i], 0]
        return
    
    def set_randomly_sampled_shape_pose(animat_options):
        for i, elem in enumerate(animat_options.morphology.joints):
            elem.initial = [random.uniform(-0.4, 0.4), 0]
        return
    
    def set_randomly_sampled_shape_pose_vel(animat_options):
        for i, elem in enumerate(animat_options.morphology.joints):
            elem.initial = [random.uniform(-0.4, 0.4), random.uniform(-0.0261799, 0.02617)] # +/- 22°, +/- 1.5°/s
        return

    @staticmethod
    def set_random_shape_pose(animat_options):
        random_state = random.choice(RobotInitialState._robot_state_list)
        random_pose = random.choice(RobotInitialState._robot_pose_list)
        # print(
        #     "[set_random_shape_pose] random_state: {}, random_pose: {}".format(
        #         random_state, random_pose
        #     )
        # )
        RobotInitialState.set_shape_and_pose_static(
            animat_options=animat_options, shape=random_state, pose=random_pose
        )
        return

    def setup(self, animat_options):
        self.set_shape_and_pose(animat_options)


if __name__ == "__main__":
    raise ValueError("Not a file that is supposed to be run")
