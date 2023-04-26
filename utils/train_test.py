import os
import pickle
from enum import Enum

from rlgym.rl_gym import FarmsGym, GymTestCallback
# from simulation import *  #setup_simulation
# import simulation as engine  #setup_simulation 
# from simulation import 
from . import simulation as engine  #setup_simulation
from rlgym.rl_gym import ActionChoice, ObservationChoice

from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

class TrainTestOption(Enum):
    TRAIN = 0
    TEST = 1
    CONT = 2

class TrainTestClass():
    """ TrainTestClass

    @brief: TrainTestClass is a class that helps in training, testing and continuing training of RL models.It is designed to be used with the
    FarmsGym environment for training.

    @details: TrainTestClass is a wrapper around the Stable Baselines3 library and OpenAI Gym. It brings together all the components required for
    training, testing and continuing training of RL models. The usual flow of the TrainTestClass is as follows:
        - Create the TrainTestClass object with options for FARMS, Gym Environment, RL model
        - Call the exp_training() method to train the model. This method in turn:
            - creates a simulation object with mujoco through FARMS options 
            - creates the FarmsGym environment(Gym Environment) with the simulation object
            - creates the RL model using Stable Baselines3 library
            - trains the model using the FarmsGym environment
        - Call the exp_train_cont() method to continue training the model if needed
        - Call the exp_testing() method to test the model
            - Loads the trained model 
            - Creates callback from GymTestCallback class. This callback contains the model and is used to query for action via callbacks. 
            - Creates the FARMS simulation with mujoco and provides the above callback
    
    Training is done with help of the FarmsGym class. 
    This class is an openAI Gym environment that uses FARMS as the backend simulator framework. As OpenAI gym requires control for running each 
    step of the simulation, the TrainTestClass is used to create the environment and simulation(mujoco through farms) and provide the control to the 
    learning environment.

    Testing is performed via GymTestCallback. For testing we use FARMS in its default mode. The GymTestCallback is provided as the callback 
    class to the FarmsGym environment. The callback is called at each step. At every step the observation is passed to the model and the action 
    given by the model is applied. 

    """
    def __init__(self, animat_options, arena_options, sim_options, simulator, log_dir, action_choice: ActionChoice, observation_choice:ObservationChoice, learn_total_timesteps: int):
        """Constructor for TrainTestClass
        
        Responsibility: train, continue training and testing 

        Args:
            animat_options (_type_): animat_options
            arena_options (_type_): arena_options
            sim_options (_type_): simulation options
            simulator (_type_): simulator choice
        """
        self.animat_options = animat_options
        self.arena_options = arena_options
        self.sim_options = sim_options
        self.simulator = simulator
        self.log_dir = log_dir
        self.save_test_data = False # TODO: setup kwargs option
        self.action_choice = action_choice
        self.observation_choice = observation_choice
        self.learn_total_timesteps = learn_total_timesteps

    def exp_train_cont(self, model_filename_original: str, model_filename_to_cont:str) -> None:
        """Continue experiment training on saved models
        
        @note: original model and continued model are saved in the same directory. The directory is named according to the 
        date of the original model.

        @param model_filename_original (str): Name of the original saved model.
        @param model_filename_to_cont (str): Name of the further trained model.
        """

        # setup simulation
        sim, animat_data = engine.setup_simulation(self.animat_options, self.arena_options, self.sim_options, self.simulator, callbacks=[])

        # setup gym: create the environment with mujoco sim
        gym_env = FarmsGym(
            timestep=self.sim_options.timestep,
            observation_choice=self.observation_choice,
            action_choice=self.action_choice,
            sim=sim,
        )

        # load semi-trained model
        model_log_dir = os.path.join(self.log_dir, '..', model_filename_original)
        model = RecurrentPPO.load(os.path.join(model_log_dir, model_filename_original), env=gym_env)
        
        # configure logger
        new_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
        callback = CheckPointCallbacks.callbacks_list(log_dir=self.log_dir)
        model.set_logger(new_logger)
        
        # continue training
        model.learn(total_timesteps=self.learn_total_timesteps, callback=callback)
        model.save(os.path.join(self.log_dir, model_filename_to_cont))


    def exp_training(self, model_filename:str) -> None:
        """Experiment training

        @param model_filename (str): Name of the saved model.
        """

        # setup simulation
        sim, animat_data = engine.setup_simulation(self.animat_options, self.arena_options, self.sim_options, self.simulator, callbacks=[])

        # setup gym: create the environment with mujoco sim
        gym_env = FarmsGym(
            timestep=self.sim_options.timestep,
            observation_choice=self.observation_choice,
            action_choice=self.action_choice,
            sim=sim,
        )

        # Create the Model
        model = RecurrentPPO(
            "MlpLstmPolicy",
            gym_env,
            n_steps=1000,
            batch_size=500,
            verbose=2,
            # ent_coef=0.3,
            # learning_rate=0.03,
            clip_range=0.5,
            #  clip_range_vf=20,
            # gamma=0.99,
            tensorboard_log=self.log_dir, 
            )

        # configure logger
        new_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
        callback = CheckPointCallbacks.callbacks_list(log_dir=self.log_dir)
        model.set_logger(new_logger)

        # train
        model.learn(total_timesteps=self.learn_total_timesteps, callback=callback)
        model.save(os.path.join(str(self.log_dir), str(model_filename)))

    def exp_testing(self, model_filename:str, debug_random_cond: bool)-> None:
        """Experiment testing 

        @param model_filename (str): Name of the saved model.
        @param debug_random_cond (bool): If true, the animat is tested in random conditions.
        """
        # load trained model
        model = RecurrentPPO.load(os.path.join(self.log_dir, model_filename))
        
        # callback on trained model for testing
        gymTestCallback = GymTestCallback(
            timestep=self.sim_options.timestep,
            n_iterations=self.sim_options.n_iterations,
            model=model,
            observation_choice=self.observation_choice,
            action_choice=self.action_choice,
            debug_random_cond=debug_random_cond,
        )
        callbacks = [ gymTestCallback ]
        
        # setup simulation
        sim, animat_data = engine.setup_simulation(self.animat_options, self.arena_options, self.sim_options, self.simulator, callbacks=callbacks)
        # setup callback for testing model 
        gymTestCallback.set_mujoco_model(sim)
        # run simulation
        sim.run()

        if self.save_test_data:
            self.exp_save_run(sim, animat_data)
        return

    def exp_save_run(self, sim, animat_data)-> None:
        """Save simulation data
        
        @param sim (_type_): simulation object
        @param animat_data (_type_): animat data object
        """
        filename = '{}/test_simulation.{}'
        animat_data.to_file(filename.format(self.log_dir, 'h5'), sim.iteration)
        with open(filename.format(self.log_dir, 'pickle'), 'wb') as param_file:
            pickle.dump(self.sim_options, param_file)
        print(filename.format(self.log_dir, 'h5'))
        print(filename.format(self.log_dir, 'pickle'))
        return

    def arch_testing(self)-> None:
        """Test the architecture of farms

        @brief: This function is used to check the options for FARMS and Notions(ExperimentOptions)
        """
        callbacks = []
        sim, _ = engine.setup_simulation(self.animat_options, self.arena_options, self.sim_options, self.simulator, callbacks=callbacks)
        sim.run()
        return


class CheckPointCallbacks:

    @staticmethod
    def callbacks_list(log_dir):
        """ Create and retun checkpoint callback for gym training

        Args:
            log_dir (string): abs path for log dir

        Returns:
            list: list containing checkpoint callback
        """
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir)
        callback = CallbackList([checkpoint_callback])

        return callback

