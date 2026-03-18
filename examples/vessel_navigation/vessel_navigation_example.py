import numpy as np
import copy
import time
import os
import enum

from pacSTL.pacSTL_utils import PACSignalTemporalLogic
from reachability_utils.trigonometry_utils import rotation_matrix
from examples.evaluate_reachable_sets import get_reachable_sets
from vessel_utils import  T_H, R_EGO, R_EGO_DRILL, S_DOT_MAX_DRILL, InFrontRobustness, CollisionRobustness
from vessel_utils import step_sim, DrillshipSimulator
import pickle

ros_dict = {
    "ego": {
        "p_x": 7.0,
        "p_y": -1.0,
        "v_x": -0.09 *2,
        "v_y": 0.026*2,
        "psi": 2.89
    },
    "other": {
        "p_x": -4.0,
        "p_y": 1.5,
        "v_x": 0.33,
        "v_y": 0.0,
        "psi": 5.812238456841166,
        "u": 0.09
    },
    "initial_other": {
            "p_x": -4.0,
            "p_y": 1.5,
            "psi": 5.812238456841166,
        }
}

PRED_HORIZON = 5  # in DT
DT = 0.5  # in s
MIN_TURNING_ANGLE = -0.8  # in rad
TIME_TURNING = DT * 45  # in s
TIME_PARALLEL = DT * 30  # in s



def transform_and_pred(state_ego, translation, rotation):
    pred_states_ego_in_other_frame = {}

    # transform current ego state
    pos_ego_tank = np.array([-state_ego["p_x"], state_ego["p_y"]])
    vel_ego_tank = np.array([-state_ego["v_x"], state_ego["v_y"]])
    pos_ego_other = rotation_matrix(rotation + np.pi, radian=True) @ (pos_ego_tank + translation)
    vel_ego_other = rotation_matrix(rotation + np.pi, radian=True) @ vel_ego_tank
    head = -state_ego["psi"] + rotation
    abs_vel = np.linalg.norm(np.array([state_ego["v_x"], state_ego["v_y"]]))

    state_ego_array = np.array([pos_ego_other[0], pos_ego_other[1], head, vel_ego_other[0], vel_ego_other[1], abs_vel])
    pred_states_ego_in_other_frame[0] = state_ego_array

    # prediction
    heading_vec = np.array([np.cos(state_ego_array[2]), np.sin(state_ego_array[2])])
    distance_advance_ego = heading_vec * state_ego_array[5]

    i = 1
    state_ego_pred = copy.deepcopy(state_ego_array)
    while i <= PRED_HORIZON:
        state_ego_pred[0:2] = state_ego_array[0:2] + distance_advance_ego * DT * i
        pred_states_ego_in_other_frame[i] = copy.deepcopy(state_ego_pred)
        i += 1

    return pred_states_ego_in_other_frame


def get_wp(state_ego):
    heading_1 = state_ego["psi"] - MIN_TURNING_ANGLE
    heading_vec_1 = np.array([np.cos(heading_1), np.sin(heading_1)])
    turning_wp = np.array([state_ego["p_x"], state_ego["p_y"]]) + np.linalg.norm(
        np.array([state_ego["v_x"], state_ego["v_y"]])) * heading_vec_1 * TIME_TURNING

    heading_vec_2 = np.array([np.cos(state_ego["psi"]), np.sin(state_ego["psi"])])
    parallel_wp = turning_wp + np.linalg.norm(
        np.array([state_ego["v_x"], state_ego["v_y"]])) * heading_vec_2 * TIME_PARALLEL

    return [turning_wp, parallel_wp]


def thread_atomic_pred(func, state_ego_array, ellipsoid_A, ellipsoid_b, center, dict, key):
    value = func(state_ego_array, ellipsoid_A, ellipsoid_b, center)
    dict[key] = value


def get_atomic_intervals(pred_states_ego_in_other_frame, ellipsoids_Ab_dict, robustness_fun_dict):
    atomic_interval_dict = {}

    i = 1
    while i <= PRED_HORIZON:
        atomic_interval_dict[i] = {}
        atomic_interval_dict[i]['FrontLeft'] = robustness_fun_dict['FrontLeft'](pred_states_ego_in_other_frame[i],
                                                                                ellipsoids_Ab_dict[i][0],
                                                                                ellipsoids_Ab_dict[i][1],
                                                                                ellipsoids_Ab_dict[i][2], i)

        atomic_interval_dict[i]['Coll'] = robustness_fun_dict['Coll'](
            pred_states_ego_in_other_frame[i],
            ellipsoids_Ab_dict[i][0],
            ellipsoids_Ab_dict[i][1], ellipsoids_Ab_dict[i][2], i)

        i += 1

    return atomic_interval_dict


def get_ISTL_persistent_encounter(atomic_interval_dict):
    predicates = {}
    predicates['headon'] = {}
    predicates['crossing'] = {}

    # make predicates for temporal op
    timestep = 1
    while timestep <= PRED_HORIZON:
        ai_at_t = atomic_interval_dict[timestep]
        predicates['headon'][timestep] = PACSignalTemporalLogic.conjunction(
            {0: ai_at_t['FrontLeft'], 1: ai_at_t['Coll']})

        timestep += 1
    # temporal operation evaluation
    headon_persistent_encounter = PACSignalTemporalLogic.globally(predicates['headon'], [1, 2, 3, 4, 5])

    return headon_persistent_encounter


def pre_script():

    robustness_fun_dict = {}
    robustness_fun_dict['FrontLeft'] = InFrontRobustness()
    robustness_fun_dict['Coll'] = CollisionRobustness(t_h=T_H, r_ego=R_EGO)
    ellipsoids_Ab_dict = get_reachable_sets(os.path.join(os.path.dirname(__file__)) + "/reachable_sets_vessel.pkl")


    return ellipsoids_Ab_dict, robustness_fun_dict


def main(ros_dict):
    done = False

    i = 0
    logging_dict = {}

    ellipsoids_Ab_dict, robustness_fun_dict = pre_script()
    waypoints = [np.array([-4., 1.5])]  # in tank cosy
    maneuvering = False
    simulator = DrillshipSimulator()



    while not done:
        state_other = ros_dict["other"]
        state_ego = ros_dict["ego"]

        pred_states_ego_in_other_frame = transform_and_pred(state_ego=state_ego, translation=-np.array(
            [-state_other["p_x"], state_other["p_y"]]), rotation=state_other["psi"])

        logging_dict[i] = {"step": i, "ego": copy.deepcopy(state_ego), "other": copy.deepcopy(state_other), "encounter": False}


        # compute atomic predicate intervals
        start = time.perf_counter()
        atomic_interval_dict = get_atomic_intervals(pred_states_ego_in_other_frame, ellipsoids_Ab_dict,
                                                    robustness_fun_dict)
        # compute specification intervals for persistent encounter
        robustness_head = get_ISTL_persistent_encounter(atomic_interval_dict)
        end = time.perf_counter()
        print(f"Elapsed: {end - start:.6f} sec")

        logging_dict[i]['runtime'] = end - start
        logging_dict[i]['robustness'] = [robustness_head.low, robustness_head.high, robustness_head.t_low, robustness_head.t_high]


        # waypoint generation in tank cosy directly
        if (robustness_head.high > 0) and not maneuvering: ## make the following optimistic semantics: (robustness_head.low > 0) and not maneuvering:
            waypoints_man = get_wp(state_ego)
            waypoints_man.extend(waypoints)
            waypoints = waypoints_man
            maneuvering = True
            logging_dict[i]['encounter'] = [True, waypoints]


        if i % 5 == 0:
            # random tau every 5 steps
            # forces x, y, z, moments x, y, z
            tau_cmd = np.array(
                [np.random.uniform(0.7, 1.2), np.random.uniform(-0.1, 0.1), 0, 0, 0, np.random.uniform(-0.1, 0.1)],
                dtype=float).reshape(-1, 1)

        done, waypoints, ros_dict = step_sim(waypoints, ros_dict, simulator, tau_cmd)

        i += 1

    return logging_dict

if __name__ == '__main__':

    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)

    logs = main(ros_dict)
    file = os.path.join(script_directory, 'vessel_example_traces.pkl')
    with open(file, 'wb') as f:
        pickle.dump(logs, f)