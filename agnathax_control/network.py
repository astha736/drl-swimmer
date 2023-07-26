"""Network"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy import integrate
from scipy.integrate._ode import ode as ODE

from .ode import ode_oscillators_sparse

import conf


class AnimatNetwork(ABC):
    """Animat network"""

    def __init__(self, data, n_iterations):
        super().__init__()
        self.data: AnimatData = data
        self.n_iterations = n_iterations

    @abstractmethod
    def step(
        self,
        iteration: int,
        time: float,
        timestep: float,
        **kwargs,
    ):
        """Step function called at each simulation iteration"""


class NetworkODETEST(AnimatNetwork):
    """NetworkODE"""

    def __init__(self, data, **kwargs):
        self.state_array_init = data.state.array
        super().__init__(data=data, n_iterations=np.shape(data.state.array)[0])
        self.dstate = np.zeros_like(data.state.array[0, :])
        self.ode: Callable = kwargs.pop("ode", ode_oscillators_sparse)
        self.solver: ODE = integrate.ode(f=self.ode)
        nsteps = 1000
        if "integrator_nsteps" in conf.CONF["config"]:
            nsteps = conf.CONF["config"]["integrator_nsteps"]
        integrator = "dopri5"
        if "integrator" in conf.CONF["config"]:
            integrator = conf.CONF["config"][
                "integrator"
            ]  # dop853 seems to work best for 'drive'
        self.solver.set_integrator(integrator, nsteps=nsteps, **kwargs)
        self.solver.set_initial_value(y=data.state.array[0, :], t=0.0)

    def __reset(self):
        # sets initial values of oscillators
        self.dstate = np.zeros_like(self.state_array_init[0, :])  # reset dstate
        if conf.CONF["RL"]["useRandStartCondPhases"]:
            self.__set_random_initial_oscillator_states()
            self.solver.set_initial_value(y=self.data.state.array[0, :], t=0.0)
        else:
            self.solver.set_initial_value(
                y=self.state_array_init[0, :], t=0.0
            )  # reset initial values to initial states
        return

    def __set_random_initial_oscillator_states(self):
        if conf.CONF["RL"]["useRandStartCondPhases"] == 1:
            # perfect init cond
            initial_phase_l = np.linspace(2 * np.pi, 0, 10)
        elif conf.CONF["RL"]["useRandStartCondPhases"] == 2:
            # perfect init cond + noise
            # DEFAULT
            initial_phase_l = np.linspace(2 * np.pi, 0, 10) + np.random.uniform(
                -0.3, 0.3, size=10
            )
        elif conf.CONF["RL"]["useRandStartCondPhases"] == 3:
            # sample in [0, 2pi]
            initial_phase_l = np.random.uniform(0.0, 1.0, size=10) * 2 * np.pi
        elif conf.CONF["RL"]["useRandStartCondPhases"] == 4:
            # perfect init cond + more noise
            initial_phase_l = np.linspace(2 * np.pi, 0, 10) + np.random.uniform(
                -1.0, 1.0, size=10
            )
        elif conf.CONF["RL"]["useRandStartCondPhases"] == 5:
            # perfect init cond + even more noise
            initial_phase_l = np.linspace(2 * np.pi, 0, 10) + np.random.uniform(
                -2.0, 2.0, size=10
            )
        initial_phase_r = initial_phase_l - np.pi  # np.linspace(np.pi, -np.pi, 10)

        # set states
        for j, osci in enumerate(range(0, 20, 2)):
            self.data.state.array[0, osci] = initial_phase_l[j]
            self.data.state.array[0, osci + 1] = initial_phase_r[j]

    def copy_next_drive(self, iteration):
        """Set initial drive"""
        array = self.data.network.drives.array
        array[iteration + 1] = array[iteration]

    def step(
        self,
        iteration: int,
        time: float,
        timestep: float,
        checks: bool = False,
    ):
        """Control step"""
        if iteration == 0:
            self.__reset()
            self.copy_next_drive(iteration)
            return
        if checks:
            assert np.array_equal(self.solver.y, self.data.state.array[iteration, :])
        self.solver.set_f_params(self.dstate, iteration, self.data)
        self.data.state.array[iteration, :] = self.solver.integrate(
            time + timestep, step=True
        )
        if iteration < self.n_iterations - 1:
            self.copy_next_drive(iteration)
        if checks:
            assert self.solver.successful()
            assert abs(time + timestep - self.solver.t) < 1e-6 * timestep, (
                "ODE solver time: "
                f"{self.solver.t} [s] != Simulation time: {time+timestep} [s]"
            )
