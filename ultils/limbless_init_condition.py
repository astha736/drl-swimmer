import numpy as np

class LimblessExperimentInitialization:
    """Get the options for Initialization"""

    def __init__(self, condition_shape, pose):
        self.condition_shape = condition_shape
        self.pose  = pose

    def set_initial_conditions_Lshape(animat_options, pose=0):
        """ Initial condition with L shape
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
                joint_initial_pos = 0.0 # -(np.pi/7)
            else:
                # head i = 0, 1, 2, 3 (head)
                joint_initial_pos = -(np.pi/8)
            elem.initial = [joint_initial_pos , 0]


        # # shape of the robot
        # for i, elem in enumerate(animat_options.morphology.joints):
        #     if i > 5:
        #         # tail i = 7, 8, 9 (tail)
        #         joint_initial_pos = (np.pi/8)
        #     elif i > 3:
        #         # trunk i = 4, 5, 6 (trunk) 
        #         joint_initial_pos = 0.0 # -(np.pi/7)
        #     else:
        #         # head i = 0, 1, 2, 3 (head)
        #         joint_initial_pos = 0 #-(np.pi/8)
        #     elem.initial = [joint_initial_pos , 0]
        
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
        """ Initial condition with S shape
            Shape of the robot is defined with the for loop
            Pose of the robot is defined as [x, y, z, r_x, r_y, r_z]

            Note: Aim of these initial  condition is to avoid collision with pegs
            when launched
        """
        # robot shape
        for i, elem in enumerate(animat_options.morphology.joints):
            if i > 5:
                # tail i = 6, 7, 8, 9 (tail)
                joint_initial_pos = (np.pi/8)
            elif i > 3:
                # trunk i = 4, 5 (trunk) 
                joint_initial_pos = 0.0 # -(np.pi/7)
            else:
                # head i = 0, 1, 2, 3 (head)
                joint_initial_pos = -(np.pi/8)
            elem.initial = [joint_initial_pos , 0]
        
        # robot pose
        if pose == 0:
            animat_options.spawn.pose = [ -0.08, -0.00, 0.1, 0.0, 0.0, 0.0]
        elif pose == 1:
            animat_options.spawn.pose = [-0.000, -0.06, 0.1, 0.0, 0.0, 0.3]
        elif pose == 2:
            animat_options.spawn.pose = [-0.000, -0.06, 0.1, 0.0, 0.0, 0.2]
        else:
            raise ValueError("Case for desired pose option not specified")
        return

    def set_initial_conditions_Ushape(animat_options, pose=0):
        """ Initial condition with U shape
            Shape of the robot is defined with the for loop
            Pose of the robot is defined as [x, y, z, r_x, r_y, r_z]

            Note: Aim of these initial  condition is to avoid collision with pegs
            when launched
        """
        # robot shape
        for i, elem in enumerate(animat_options.morphology.joints):
            if i > 4:
                # tail i = 5, 6, 7, 8, 9 (tail)
                joint_initial_pos = -(np.pi/8)
            else:
                # head i = 0, 1, 2, 3, 4 (head)
                joint_initial_pos = -(np.pi/8)
            elem.initial = [joint_initial_pos , 0]

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
        """ Initial condition with Line shape, but placed parallel to the pegs
            Shape of the robot is defined with the for loop
            Pose of the robot is defined as [x, y, z, r_x, r_y, r_z]

            Note: Aim of these initial  condition is to avoid collision with pegs
            when launched
        """
        # robot shape
        for i, elem in enumerate(animat_options.morphology.joints):
            elem.initial = [0.0 , 0]
        
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
        """ Initial condition with Line shape, but placed diagonal to the pegs
            Shape of the robot is defined with the for loop
            Pose of the robot is defined as [x, y, z, r_x, r_y, r_z]

            Note: Aim of these initial  condition is to avoid collision with pegs
            when launched
        """
        # robot shape
        for i, elem in enumerate(animat_options.morphology.joints):
            elem.initial = [0.0 , 0]

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
    
    def set_shape_and_pose(self, animat_options):

        if self.condition_shape == 'Lshape':
            LimblessExperimentInitialization.set_initial_conditions_Lshape(animat_options, self.pose)
        elif self.condition_shape == 'Sshape':
            LimblessExperimentInitialization.set_initial_conditions_Sshape(animat_options, self.pose)
        elif self.condition_shape == 'Ushape':
            LimblessExperimentInitialization.set_initial_conditions_Ushape(animat_options, self.pose)
        elif self.condition_shape == 'Parallel':
            LimblessExperimentInitialization.set_initial_conditions_parallel(animat_options, self.pose)
        elif self.condition_shape == 'Diagonal':
            LimblessExperimentInitialization.set_initial_conditions_diagonal(animat_options, self.pose)
        else:
            raise ValueError(' condition_shape value {}, not found. Please check again'.format(self.condition_shape))

    def setup(self, animat_options):
        self.set_shape_and_pose(animat_options)
