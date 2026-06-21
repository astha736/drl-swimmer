#!/usr/bin/env python3
"""Simulation utils for gym environment"""
# import os
# from datetime import date
from re import X
from typing import Union
import numpy as np


from farms_core import pylog
from farms_core.simulation.options import Simulator
from farms_mujoco.simulation.simulation import Simulation as MuJoCoSimulation

# from farms_sim.utils.parse_args import sim_parse_args
from farms_sim.simulation import (
    simulation_setup,
)

from farms_amphibious.callbacks import setup_callbacks
from farms_amphibious.data.data import (
    AmphibiousData,
    AmphibiousKinematicsData,
    get_amphibious_data,
)


from agnathax_control.network import NetworkODETEST
from farms_amphibious.control.kinematics import KinematicsController
from farms_amphibious.control.amphibious import (
    AmphibiousController,
    get_amphibious_controller,
)

from farms_core.model.options import AnimatOptions, ArenaOptions
from .camera import CameraCallback, save_video

ENGINE_BULLET = False
try:
    from farms_amphibious.bullet.simulation import (
        AmphibiousPybulletSimulation,
        pybullet_simulation_kwargs,
    )

    ENGINE_BULLET = True
except ImportError as err:
    AmphibiousPybulletSimulation = None
    pybullet_simulation_kwargs = None


def create_simulation(
    animat_options: AnimatOptions,
    arena_options: ArenaOptions,
    **kwargs,
) -> Union[MuJoCoSimulation, AmphibiousPybulletSimulation]:
    """Create simulation object. Called by setup_simulation below.

    Args:
        animat_options (AnimatOptions): animat_options
        arena_options (ArenaOptions): arena_options

    options are usually created by setup_from_clargs()
        parameters
        (
            clargs=clargs,                              # setup from user
            animat_options_loader=AmphibiousOptions,    # passing class which is used to load
            arena_options_loader=AmphibiousArenaOptions,# passing class which is used to load
        )

    Returns:
        Union[MuJoCoSimulation, AmphibiousPybulletSimulation]: either object for MUJOCO or PYBULLET simulation
                    created and passed as output
    """
    # Instatiate simulation
    # pylog.info("Creating simulation")
    simulator = kwargs.get("simulator", Simulator.MUJOCO)
    sim = simulation_setup(animat_options, arena_options, **kwargs)
    return sim


def setup_simulation(animat_options, arena_options, sim_options, simulator, callbacks):
    """setup the simulation with agnathax_control network and odes

    Args:
        animat_options (AnimatOptions): animat options variable from class
        arena_options (ArenaOptions): arena options variable from class
        sim_options (SimulationOptions): simulation option variables from class
        simulator (Simulator): Simulator variable. e.g. Simulator.MUJOCO
        callbacks (list): list of callbacks for simulation
    Returns:
        sim: simulation object
        animat_data: animat data object
    """
    # Data
    animat_data: Union[AmphibiousData, AmphibiousKinematicsData] = get_amphibious_data(
        animat_options=animat_options,
        simulation_options=sim_options,
    )

    # Network
    if isinstance(animat_data, AmphibiousData):
        animat_network = NetworkODETEST(animat_data, max_step=sim_options.timestep)
        controller_args = {"animat_network": animat_network}
    else:
        controller_args = {}

    # Controller
    animat_controller: Union[
        AmphibiousController, KinematicsController
    ] = get_amphibious_controller(
        animat_data=animat_data,
        animat_options=animat_options,
        sim_options=sim_options,
        **controller_args,
    )

    # Additional engine-specific options
    options = {}
    camera = None
    if simulator == Simulator.MUJOCO:
        if sim_options.video:
            camera = CameraCallback(
                camera_id=0,
                timestep=sim_options.timestep,
                n_iterations=sim_options.n_iterations,
                fps=sim_options.video_fps,
                width=sim_options.video_resolution[0],
                height=sim_options.video_resolution[1],
            )
        options['callbacks'] = setup_callbacks(
            animat_options=animat_options,
            arena_options=arena_options,
            camera=camera,
        )
    elif simulator == Simulator.PYBULLET:
        raise NotImplementedError
    
    # Simulation
    # pylog.info("Creating simulation environment")
    sim: Union[MuJoCoSimulation, AmphibiousPybulletSimulation] = create_simulation(
        animat_data=animat_data,
        animat_options=animat_options,
        animat_controller=animat_controller,
        simulation_options=sim_options,
        arena_options=arena_options,
        simulator=simulator,
        **options,
    )

    return sim, animat_data
