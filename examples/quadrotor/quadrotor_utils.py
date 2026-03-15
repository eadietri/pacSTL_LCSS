import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy.random import default_rng

from pacSTL.pacSTL_utils import EllipsoidalSignalTemporalLogic
from pacSTL.atomic_robustness_bounds import Robustness, Predicate

# Dynamics sampler -- https://easychair.org/publications/paper/gjfh/open contains c

GRAVITY = 9.81
M = 1
M_ROTOR = 4 *0.1
L = 0.5
R = 0.1
MASS = M + M_ROTOR#
INERTIA_X = 0.4 * M * (R**2) +  2 * M_ROTOR *(L**2)
INERTIA_Y = 0.4 * M * (R**2) +  2 * M_ROTOR *(L**2)
INERTIA_Z = 0.4 * M * (R**2) +  4 * M_ROTOR *(L**2)
H_REF = 1
T_FINAL = 5
DT = 0.25 # potentially refine more
MIN_X0 = -0.2
MAX_X0 = 0.2



def quadrotor(y, t, noise_series):

    pn, pe, h, phi, theta, psi, pn_v, pe_v, h_v, phi_v, theta_v, psi_v = y

    noise = noise_series

    F_calc = MASS * GRAVITY - 10 * (h - H_REF) - 3 * h_v
    F = max(0.5 * MASS * GRAVITY, min(F_calc, 1.5 * MASS * GRAVITY))

    tau_phi = -phi - phi_v
    tau_theta = -theta - theta_v
    tau_psi = 0

    dydt = [
        pn_v, pe_v, h_v, phi_v, theta_v, psi_v,

        F/MASS * (-np.cos(phi)*np.sin(theta)*np.cos(psi)
                  -np.sin(phi)*np.sin(psi)),

        F/MASS * (-np.cos(phi)*np.sin(theta)*np.sin(psi)
                  +np.sin(phi)*np.cos(psi)),

        F/MASS * (np.cos(phi)*np.cos(theta)) - GRAVITY,

        tau_phi/INERTIA_X + noise[0],
        tau_theta/INERTIA_Y + noise[1],
        tau_psi/INERTIA_Z + noise[2]
    ]

    return dydt


def sample_quadrotor():
    rn = default_rng().normal
    mu, sigma = 0, 0.15
    y0 = np.zeros(12)
    y0[[0, 1, 2, 6, 7, 8]] = rn(mu, sigma, 6) #np.random.uniform(MIN_X0, MAX_X0, 6)
    mu_noise, sigma_noise = 0, 0.02
    y0[[3, 4, 5]] = rn(mu_noise, sigma_noise, 3)
    t = np.linspace(DT, T_FINAL, int(T_FINAL/DT))
    noise_series = rn(mu_noise, sigma_noise, 3)
    reach = odeint(quadrotor, y0, t, args=(noise_series,))

    # --- Plotting Code ---
    # plt.figure(figsize=(8, 5))
    # plt.plot(t, reach[:, 2], label='Dim 3', color='blue')

    # plt.title('Quadrotor h over Time')
    # plt.xlabel('Time')
    # plt.ylabel('State Value')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # ---------------------

    return reach

def make_quadrotor_samples(ndata: int):
    samples = []
    for i in range(ndata):
        samples.append(sample_quadrotor())

    return np.array(samples)

# atomic propositions

class LinearComparison(Predicate):

    def __init__(
        self,
        threshold: float = 0.0,
        A: np.array = None,
    ):
        self.threshold = threshold
        self.A = A

    def _evaluate(
        self,
        state: np.ndarray,
    ) -> float:

        h = self.A @ state - self.threshold

        return h

class QuadraticComparison(Predicate):

    def __init__(
        self,
        threshold: float = 0.0,
        Q: np.array = None,
        Q_factor: float = 0.0
    ):
        self.threshold = threshold
        self.Q = Q
        self.Q_factor = Q_factor

    def _evaluate(
        self,
        state: np.ndarray,
    ) -> float:

        h = self.Q_factor * (np.sum( self.Q * (state**2)) - self.threshold)

        return h

class HeightLater(Robustness):
    def __init__(
        self,
        h_low: float = 0.0,
        h_high: float = 0.0,
        threshold: float = 0.9,
        A: np.array = np.array([0, 0, 1, 0,0,0, 0,0,0, 0,0,0]),
    ):
        predicate = LinearComparison(threshold=threshold, A=A)
        super().__init__(pred=predicate)

    def compute_robustness(self, ellipsoid_A, ellipsoid_b, center, time_step):
        min_h = self.min_linear_predicates(self.pred.A, self.pred.threshold, ellipsoid_A, ellipsoid_b,center)
        max_h = self.max_linear_predicates(self.pred.A, self.pred.threshold, ellipsoid_A, ellipsoid_b,center)

        return EllipsoidalSignalTemporalLogic(min(min_h, max_h), max(min_h, max_h), time_step, time_step)

    def __call__(self, ellipsoid_A, ellipsoid_b,center, time_step):
        return self.compute_robustness(ellipsoid_A, ellipsoid_b,center, time_step)

class HeightAlways(Robustness):
    def __init__(
        self,
        h_low: float = 0.0,
        h_high: float = 0.0,
        threshold: float = - 1.5,
        A: np.array = np.array([0, 0, - 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ):
        predicate = LinearComparison(threshold=threshold, A=A)
        super().__init__(pred=predicate)

    def compute_robustness(self, ellipsoid_A, ellipsoid_b, center, time_step):
        min_h = self.min_linear_predicates(self.pred.A, self.pred.threshold, ellipsoid_A, ellipsoid_b,center)
        max_h = self.max_linear_predicates(self.pred.A, self.pred.threshold, ellipsoid_A, ellipsoid_b,center)

        return EllipsoidalSignalTemporalLogic(min(min_h, max_h), max(min_h, max_h), time_step, time_step)

    def __call__(self, ellipsoid_A, ellipsoid_b,center, time_step):
        return self.compute_robustness(ellipsoid_A, ellipsoid_b,center, time_step)

# TODO: double check these implementations for quadratic atomic propositions to make sure the underlying functions can find global min/max
class VelocityBound(Robustness):
    def __init__(
        self,
        h_low: float = 0.0,
        h_high: float = 0.0,
        threshold: float = 1.0,
        Q: np.array = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]),
        Q_factor: float = - 1.0,
    ):
        predicate= QuadraticComparison(threshold=threshold, Q=Q, Q_factor=Q_factor)
        super().__init__(pred=predicate)

    def compute_robustness(self, ellipsoid_A, ellipsoid_b, center, time_step):
        temp_1 = - self.min_quadratic_predicates(self.pred.Q, self.pred.threshold, self.pred.Q_factor, ellipsoid_A, ellipsoid_b,center)
        temp_2 = - self.max_quadratic_predicates_langrage(ellipsoid_A, center, [6,7,8], 1.0, self.pred.threshold)

        return EllipsoidalSignalTemporalLogic(min(temp_1, temp_2), max(temp_1, temp_2), time_step, time_step)

    def __call__(self, ellipsoid_A, ellipsoid_b,center, time_step):
        return self.compute_robustness(ellipsoid_A, ellipsoid_b,center, time_step)


class AngularVelocityBound(Robustness):
    def __init__(
        self,
        h_low: float = 0.0,
        h_high: float = 0.0,
        threshold: float = 1.0,
        Q: np.array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
        Q_factor: float = - 1.0,
    ):
        predicate = QuadraticComparison(threshold=threshold, Q=Q, Q_factor=Q_factor)
        super().__init__(pred=predicate)

    def compute_robustness(self, ellipsoid_A, ellipsoid_b, center, time_step):
        temp_1 = - self.min_quadratic_predicates(self.pred.Q, self.pred.threshold, self.pred.Q_factor, ellipsoid_A, ellipsoid_b,center)
        temp_2 = - self.max_quadratic_predicates_langrage(ellipsoid_A, center, [9,10,11], 1.0, self.pred.threshold)

        return EllipsoidalSignalTemporalLogic(min(temp_1, temp_2), max(temp_1, temp_2), time_step, time_step)

    def __call__(self, ellipsoid_A, ellipsoid_b,center, time_step):
        return self.compute_robustness(ellipsoid_A, ellipsoid_b,center, time_step)
