import numpy as np
from abc import ABC, abstractmethod

# base shoebox simulator class
class BaseSimulator(ABC):
    def __init__(self):

        self.eta0 = np.zeros((6, 1))
        self.nu0 = np.zeros((6, 1))
        self.u = None

        self.vessel = self._create_vessel()
        self.actuators = self._init_thrusters()
        self.deallocator = self._init_deallocator()
        self.deallocator.compute_configuration_matrix()

        self.dt = 0.01

    @abstractmethod
    def _create_vessel(self):
        pass

    @abstractmethod
    def _init_deallocator(self):
        pass

    @abstractmethod
    def _init_thrusters(self):
        pass

    def iterate(self, tau_ext : np.ndarray = None):
        tau = (self.allocator._b_matrix @ self.u).flatten()

        if tau_ext is not None:
            tau += tau_ext.flatten()

        self.vessel.step(tau=tau, dt=self.dt)