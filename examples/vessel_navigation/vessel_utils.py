import numpy as np
import shoeboxpy
import shoeboxpy.model6dof
import skadipy.actuator
import skadipy.allocator
from numpy.random import default_rng

from reachability_utils.trigonometry_utils import rotation_matrix, normalize_degree
from pacSTL.atomic_robustness_bounds import Predicate, Robustness
from pacSTL.pacSTL_utils import EllipsoidalSignalTemporalLogic, SignalTemporalLogic
from reachability_utils.data_utils import body_to_world
from examples.vessel_navigation.shoebox_sim import BaseSimulator

# Reachability sample generation

class DrillshipSimulator(BaseSimulator):

    def __init__(self):
        super().__init__()

        self.allocator = self._init_allocator()

    # This is where we create the vessel instance
    def _create_vessel(self):
        rng = default_rng()
        ru = rng.uniform

        # 1. Generate the random array and reshape it to (6, 1)
        random_nu = np.array([
            ru(0.3, 0.5), ru(-0.1, 0.1), ru(-0.092, 0.0111),
            ru(-0.092, 0), ru(-0.079, 0), ru(-0.1, 0.1)
        ]).reshape(6, 1)

        # 2. Update the simulator's attribute
        self.nu0 = random_nu

        # parameters for Drillship
        return shoeboxpy.model6dof.Shoebox(
            L=2.578,
            B=0.440,
            T=0.02,
            GM_theta=0.02, 
            GM_phi=0.02,
            eta0=np.zeros(6),
            # initialize nu
            nu0 = self.nu0.flatten()
        )

    def _init_thrusters(self):
        tunnel = skadipy.actuator.Fixed(
            position=skadipy.toolbox.Point([0.3875, 0.0, -0.01]),
            orientation=skadipy.toolbox.Quaternion(
                axis=(0.0, 0.0, 1.0), radians=np.pi / 2.0
            ),
        )
        port_azimuth = skadipy.actuator.Azimuth(
            position=skadipy.toolbox.Point([-0.4574, -0.055, -0.1]),
        )
        starboard_azimuth = skadipy.actuator.Azimuth(
            position=skadipy.toolbox.Point([-0.4547, 0.055, -0.1]),
        )
        # This order is important when unpacking the command vector
        return [tunnel, port_azimuth, starboard_azimuth]

    def _init_deallocator(self):
        dofs = [
            skadipy.allocator.ForceTorqueComponent.X,
            skadipy.allocator.ForceTorqueComponent.Y,
            skadipy.allocator.ForceTorqueComponent.Z,
            skadipy.allocator.ForceTorqueComponent.K,
            skadipy.allocator.ForceTorqueComponent.M,
            skadipy.allocator.ForceTorqueComponent.N,
        ]
        return skadipy.allocator.PseudoInverse(
            actuators=self.actuators, force_torque_components=dofs
        )

    def _init_allocator(self):
        allocator = skadipy.allocator.PseudoInverse(
            actuators=self.actuators,
            force_torque_components=[
                skadipy.allocator.ForceTorqueComponent.X,
                skadipy.allocator.ForceTorqueComponent.Y,
                skadipy.allocator.ForceTorqueComponent.N,
            ],
        )
        allocator.compute_configuration_matrix()
        return allocator

    def command_u(self, u):
        """
        Set the thruster command vector.
        This method is called by the allocator to set the thruster commands.
        """
        if u.shape[0] != self.allocator._b_matrix.shape[1]:
            raise ValueError(
                "Command vector dimension does not match thruster configuration."
            )
        self.u = u.reshape(-1, 1)

def compute_vessel_corners(x, y, yaw, length=1.0, width=0.3):
    # Half dimensions
    dx = length / 2
    dy = width / 2

    # Rectangle corners in body frame (relative to center)
    corners_body = np.array([
        [ dx,  dy],  # front-left
        [ dx, -dy],  # front-right
        [-dx, -dy],  # rear-right
        [-dx,  dy],  # rear-left
    ])

    # Rotation matrix for yaw (heading)
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    # Rotate and translate corners to world frame
    corners_world = (R_yaw @ corners_body.T).T + np.array([x, y])

    return corners_world


def step_vessel(simulator, tau_cmd):
    simtime = 0.51
    T = np.arange(0, simtime, simulator.dt) # simulator.dt = 0.01

    eta = np.zeros((6, len(T)))
    nu = np.zeros((6, len(T)))

    for i, t in enumerate(T):
        u = np.zeros((simulator.allocator._b_matrix.shape[1], 1))
        b = np.zeros((6, 1))
        if i % 50 == 0:
            # allocate thrust
            u, _ = simulator.allocator.allocate(tau_cmd)

            simulator.command_u(u)
        # actually advance the simulation
        simulator.iterate(tau_ext=b)

        # results from each step
        eta[:, i] = simulator.vessel.eta
        nu[:, i] = simulator.vessel.nu

    extra_values = np.array([body_to_world(body_vel_x, body_vel_y, yaw) for body_vel_x, body_vel_y, yaw in
                             zip(nu[0, :], nu[1, :], eta[5, :])])

    last_state = np.array([eta[0, -1], eta[1, -1], eta[5, -1], extra_values[-1, 0], extra_values[-1, 1]])

    return last_state

def sample_vessel():
    simulator = DrillshipSimulator()

    simtime = 2.51
    T = np.arange(0, simtime, simulator.dt)
    rng = default_rng()
    ru = rng.uniform


    eta = np.zeros((6, len(T)))
    nu = np.zeros((6, len(T)))

    vessel_corners = []
    reach = []
    # random initial tau
    # forces x, y, z, moments x, y, z
    tau_cmd = np.array([ru(0.7, 1.2),  ru(-0.1, 0.1), 0, 0, 0, ru(-0.1, 0.1)],    dtype=float).reshape(-1, 1)
    for i, t in enumerate(T):
        u = np.zeros((simulator.allocator._b_matrix.shape[1], 1))
        b = np.zeros((6, 1))
        if i % 100 == 0:
            # allocate thrust
            u, _ = simulator.allocator.allocate(tau_cmd)

            simulator.command_u(u)
        # actually advance the simulation
        simulator.iterate(tau_ext=b)

        # results from each step
        eta[:, i] = simulator.vessel.eta
        nu[:, i] = simulator.vessel.nu


    extra_values = np.array([body_to_world(body_vel_x, body_vel_y, yaw) for body_vel_x, body_vel_y, yaw in zip(nu[0, :], nu[1, :], eta[5, :])])

    # reachability calculations 
    for i, t  in enumerate(T):
        if t % .5 == 0:
            # account for size of vessel
            corners = compute_vessel_corners(eta[0, i], eta[1, i], eta[5, i])
            for c in corners:
                vessel_corners.append(
                    [c[0], c[1], eta[5, i], extra_values[i, 0], extra_values[i, 1]]
                )
            reach.append(vessel_corners)
            vessel_corners = []
    
    reach=np.array(reach)

    return reach


def make_vessel_samples(ndata):
    return np.array([sample_vessel() for i in range(ndata)])



# Atomic propositions


S_DOT_MAX = 0.4 #v_max
S_DDOT_MAX = 0.15 #a_max
D_DOT_MAX = 0.1 #yaw_dot_max
S_DOT_MAX_DRILL = 0.4 #v_max 
S_DDOT_MAX_DRILL = 0.15 #a_max 
D_DOT_MAX_DRILL = 0.8 #yaw_dot_max 
T_H = 20.0 #in s
R_EGO = 1.0 #in m
R_EGO_DRILL = 2.6 #in m

DT_SIM = 0.5

def step_mock_sim(waypoints, ros_dict, simulator, tau_cmd):

    # mock step ego
    current_wp = waypoints[0]

    #normal heading vec toward wp
    desired_heading_vec = (current_wp - np.array([ros_dict['ego']["p_x"], ros_dict['ego']["p_y"]]))/ np.linalg.norm(current_wp - np.array([ros_dict['ego']["p_x"], ros_dict['ego']["p_y"]]))
    desired_heading_ang = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
    heading_ang = np.clip(desired_heading_ang, ros_dict['ego']["psi"]- D_DOT_MAX*DT_SIM, ros_dict['ego']["psi"]+ D_DOT_MAX*DT_SIM)
    heading_vec = np.array([np.cos(heading_ang), np.sin(desired_heading_ang)])
    distance_movement_ego = DT_SIM * heading_vec* np.linalg.norm(
        np.array([ros_dict['ego']["v_x"], ros_dict['ego']["v_y"]]))


    ros_dict['ego']['p_x'] += distance_movement_ego[0]
    ros_dict['ego']['p_y'] += distance_movement_ego[1]
    ros_dict['ego']['psi'] = np.arctan2(heading_vec[1], heading_vec[0])

    done = False
    delta_wp = np.linalg.norm(current_wp - np.array([ros_dict['ego']["p_x"], ros_dict['ego']["p_y"]]))

    if delta_wp < 0.1:
        if len(waypoints) == 1:
            done = True
        else:
            waypoints = waypoints[1:]


    # mock step other
    new_state_other = step_vessel(simulator, tau_cmd)
    # rotate z down to z up
    new_state_other *= np.array([1.0, -1.0, -1.0, 1.0, -1.0])
    ros_dict['other']['p_x'] = ros_dict['initial_other']['p_x'] + new_state_other[0]
    ros_dict['other']['p_y'] = ros_dict['initial_other']['p_y'] + new_state_other[1]
    ros_dict['other']['psi'] = ros_dict['initial_other']['psi'] + new_state_other[2]
    ros_dict['other']['v_x'] = new_state_other[3]
    ros_dict['other']['v_y'] = new_state_other[4]


    return done, waypoints, ros_dict



class AtomicPredicate(Predicate):
    """
    Atomic predicate class.
    """

    def __init__(
        self,
        factor: float = 1,
        relative_to_ego: bool = True,
        semantic_type: str = "",
    ):
        super().__init__()

        """
        ID format:
        i: input semantic type
        o: output semantic type
        p: decimal point
        m: minus sign
        """

        self.factor = factor

        self.semantic_type = semantic_type
        self.relative_to_ego = relative_to_ego

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate_fn(self, *args, **kwargs):
        return self._evaluate(*args, **kwargs)


class InPositionHalfspace(AtomicPredicate):

    def __init__(
        self,
        degree: float,
        relative_to_ego: bool = True,
        semantic_type: str = "",
        reverse_side: bool = False,
        scaling: float = S_DOT_MAX,
    ):
        super().__init__(
             scaling, relative_to_ego, semantic_type
        )
        self.rotation_matrix = rotation_matrix(90 + degree)
        self.relative_to_ego = relative_to_ego
        if reverse_side:
            self.reverse_side_factor = -1
        else:
            self.reverse_side_factor = 1

    def _evaluate(
        self,
        state_ego: np.ndarray,
        state_other: np.ndarray,
    ) -> float:

        A, b = self.provide_halfspace(state_ego)

        return A @ state_other - b

    def _provide_halfspace(
        self,
        state_ego: np.ndarray,
    ) -> tuple:

        assert self.relative_to_ego # non-relative version no implemented
        cos_sin_ego = np.array([np.cos(state_ego[2]), np.sin(state_ego[2])])

        A = self.reverse_side_factor * ((self.rotation_matrix @ cos_sin_ego.T) / self.factor)
        b = self.reverse_side_factor *  ((self.rotation_matrix @ cos_sin_ego.T)  / self.factor) @ (state_ego[:2])

        A_full = np.zeros(5)
        A_full[:2] = A

        return (A_full, b)

    def provide_halfspace(self, state_ego: np.ndarray):
        return self._provide_halfspace(state_ego)


class InFrontHalfspace(InPositionHalfspace):

    def __init__(
        self,
        degree: float = 90,
        relative_to_ego: bool = True,
        semantic_type: str = "",
        reverse_side: bool = True,
        scaling: float = S_DOT_MAX,
    ):
        degree = normalize_degree(degree)
        super().__init__(
            degree, relative_to_ego, semantic_type, reverse_side, scaling
        )

class InFrontRobustness(Robustness):
    def __init__(
        self,
        h_low: float = 0.0,
        h_high: float = 0.0,
        scaling: float = S_DOT_MAX
    ):
        super().__init__(pred=InFrontHalfspace(scaling=scaling))

    def compute_robustness(self, state_ego_array, ellipsoid_A, ellipsoid_b, center, time_step):
        pred_A, pred_b = self.pred.provide_halfspace(state_ego_array)
        min_h = self.min_linear_predicates(pred_A, pred_b, ellipsoid_A, ellipsoid_b,center)
        max_h = self.max_linear_predicates(pred_A, pred_b, ellipsoid_A, ellipsoid_b,center)

        return EllipsoidalSignalTemporalLogic(min(min_h, max_h), max(min_h, max_h), time_step, time_step)

    def __call__(self, state_ego_array, ellipsoid_A, ellipsoid_b,center, time_step):
        return self.compute_robustness(state_ego_array, ellipsoid_A, ellipsoid_b,center, time_step)

class CollisionRobustness(Robustness):
    def __init__(
        self,
        t_h: float = 5.0,
        r_ego: float = 0.5,
        h_low: float = 0.0,
        h_high: float = 0.0,
    ):
        super().__init__()
        self.t_h = t_h
        self.r_ego = r_ego
        self.Q = np.array([0, 0, 0, 1, 1])

    def compute_robustness(self, state_ego_array, ellipsoid_A, ellipsoid_b, center, time_step):
        threshold = (np.linalg.norm(state_ego_array[0:2] - center[0:2]) + self.r_ego)**2 / self.t_h**2
        v_ego = np.zeros(5)
        v_ego[3:5] = state_ego_array[3:5]
        temp_1 =  self.min_quadratic_predicates(self.Q, threshold, None, ellipsoid_A, ellipsoid_b,center, v_ego)
        temp_2 =  self.max_quadratic_predicates_langrage(ellipsoid_A, center, [3,4], 1.0, threshold)

        return EllipsoidalSignalTemporalLogic(min(temp_1, temp_2), max(temp_1, temp_2), time_step, time_step)

    def __call__(self, state_ego_array, ellipsoid_A, ellipsoid_b,center, time_step):
        return self.compute_robustness(state_ego_array, ellipsoid_A, ellipsoid_b,center, time_step)

    def evaluate_for_states(self, state_ego_array, state_other_array, time_step):
        robustness =  ( np.linalg.norm(state_ego_array[3:5] - state_other_array[3:5])**2 - (np.linalg.norm(state_ego_array[0:2] - state_other_array[0:2]) + self.r_ego)**2 / self.t_h**2)

        return SignalTemporalLogic(robustness, time_step)