# make mock-sim version that monitors simpfied spec (infront and collision risk) for head-on case

import numpy as np
import copy
import time
import os
import sys
import enum

from pacSTL.pacSTL_utils import EllipsoidalSignalTemporalLogic
from reachability_utils.trigonometry_utils import normalize_radian, rotation_matrix, normalize_radian_pi
from evaluate_reachable_sets import get_reachable_sets
from vessel_utils import S_DDOT_MAX, S_DDOT_MAX_DRILL, T_H, R_EGO, R_EGO_DRILL, S_DOT_MAX_DRILL, \
    D_DOT_MAX_DRILL, InFrontRobustness, CollisionRobustness
from vessel_utils import step_mock_sim

import matplotlib.pyplot as plt


ros_dict = {
    "ego": {
        "p_x": 5.0,
        "p_y": -1.0,
        "v_x": -0.09 *4,
        "v_y": 0.026*4,
        "psi": 2.89
    },
    "other": {
        "p_x": -4.0,
        "p_y": 1.5,
        "v_x": 0.35,
        "v_y": 0.0,
        "psi": 5.812238456841166,
        "u": 0.09
    }
}

MAX_TRAJ_LENGTH = 100
PRED_HORIZON = 5  # in DT
DT_SIM = 1.0  # in s
DT = 0.5  # in s
MIN_MANEUVER_STEPS = 10
MIN_TURNING_ANGLE = -0.8  # in rad
TIME_TURNING = DT * 60  # in s
TIME_PARALLEL = DT * 30  # in s
DEBUG = True


class ConfigType(enum.Enum):
    VD = 0  # Voyager (ego), drill (other)
    VV = 1  # Voyager, voyager
    DD = 2  # Drill, Drill
    DV = 3  # Drill, voyager


CONFIGVESSELS = ConfigType.VD


def transform_and_pred(state_ego, translation, rotation):
    pred_states_ego_in_other_frame = {}

    # transform current ego state
    pos_ego_tank = np.array([-state_ego["p_x"], state_ego["p_y"]])
    vel_ego_tank = np.array([-state_ego["v_x"], state_ego["v_y"]])
    pos_ego_other = rotation_matrix(rotation + np.pi, radian=True) @ (pos_ego_tank + translation)
    vel_ego_other = rotation_matrix(rotation + np.pi, radian=True) @ vel_ego_tank
    # TODO double check rad norm
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

    # could be in parallel instead of a loop
    # todo make states dict
    # parallelize this -- all the functions and all of the timesteps
    i = 1
    while i <= PRED_HORIZON:
        atomic_interval_dict[i] = {}
        atomic_interval_dict[i]['FrontLeft'] = robustness_fun_dict['FrontLeft'](pred_states_ego_in_other_frame[i],
                                                                                ellipsoids_Ab_dict[i][0],
                                                                                ellipsoids_Ab_dict[i][1],
                                                                                ellipsoids_Ab_dict[i][2], i)
        # start = time.perf_counter()
        # --- code to measure ---
        # i = 0
        # while i <= PRED_HORIZON:
        atomic_interval_dict[i]['Coll'] = robustness_fun_dict['Coll'](
            pred_states_ego_in_other_frame[i],
            ellipsoids_Ab_dict[i][0],
            ellipsoids_Ab_dict[i][1], ellipsoids_Ab_dict[i][2], i)
        # print(atomic_interval_dict[i]['Coll'].low)
        # print(atomic_interval_dict[i]['Coll'].high)
        # end = time.perf_counter()
        # print(f"Elapsed: {end - start:.6f} sec")

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
        predicates['headon'][timestep] = EllipsoidalSignalTemporalLogic.conjunction(
            {0: ai_at_t['FrontLeft'], 1: ai_at_t['Coll']})

        timestep += 1
    # temporal operation evaluation
    headon_persistent_encounter = EllipsoidalSignalTemporalLogic.globally(predicates['headon'], [1, 2, 3, 4, 5])

    return headon_persistent_encounter


def pre_script():
    robustness_fun_dict = {}
    if CONFIGVESSELS == ConfigType.VD or CONFIGVESSELS == ConfigType.VV:
        robustness_fun_dict['FrontLeft'] = InFrontRobustness()
        robustness_fun_dict['Coll'] = CollisionRobustness(t_h=T_H, r_ego=R_EGO)
    else:
        robustness_fun_dict['FrontLeft'] = InFrontRobustness(scaling=S_DOT_MAX_DRILL)
        robustness_fun_dict['Coll'] = CollisionRobustness(t_h=T_H, r_ego=R_EGO_DRILL)


    ellipsoids_Ab_dicts = []

    if CONFIGVESSELS == ConfigType.VD or CONFIGVESSELS == ConfigType.DD:

        for i in range(4):
            ellipsoids_Ab_dicts.append(get_reachable_sets(
                os.path.join(os.path.dirname(__file__)) + "/../ellipsoids/reachable_sets_u" + str(i + 1) + ".pkl"))
    else:
        for i in range(4):
            ellipsoids_Ab_dicts.append(get_reachable_sets(
                os.path.join(os.path.dirname(__file__)) + "/../ellipsoids/voyager_reachable_set_u" + str(
                    i + 1) + ".pkl"))

    return ellipsoids_Ab_dicts, robustness_fun_dict


def main(ros_dict):
    done = False

    i = 0
    logging_dict = {}

    ellipsoids_Ab_dicts, robustness_fun_dict = pre_script()
    waypoints = [np.array([-4., 1.5])]  # in tank cosy
    maneuvering = False

    while not done:
        # transform ego states in tank world frame to other (ellipsoid) world frame
        # todo: angle wrapping needs to be checked
        state_other = ros_dict["other"]
        state_ego = ros_dict["ego"]
        # state_ego_array = np.array([state_ego["p_x"], state_ego["p_y"], state_ego["psi"], np.linalg.norm(np.array([state_ego["v_x"], state_ego["v_y"] ]))])

        if state_other['u'] < 0.1:
            ellipsoids_Ab_dict = ellipsoids_Ab_dicts[0]
        elif state_other['u'] < 0.3:
            ellipsoids_Ab_dict = ellipsoids_Ab_dicts[1]
        elif state_other['u'] < 0.5:
            ellipsoids_Ab_dict = ellipsoids_Ab_dicts[2]
        else:
            ellipsoids_Ab_dict = ellipsoids_Ab_dicts[3]

        # note: angle ego between 0 and 2pi now
        pred_states_ego_in_other_frame = transform_and_pred(state_ego=state_ego, translation=-np.array(
            [-state_other["p_x"], state_other["p_y"]]), rotation=state_other["psi"])
        print("transform check")
        print(np.linalg.norm(pred_states_ego_in_other_frame[0][0:2]) - np.linalg.norm(
            [state_other["p_x"] - state_ego["p_x"], state_other["p_y"] - state_ego["p_y"]]))
        print(abs(pred_states_ego_in_other_frame[0][2]) - abs(state_ego['psi'] - state_other['psi']))

        if DEBUG and i % 10 == 0:

            # project_plot_ellipsoid_2d(ellipsoids_Ab_dict, 0,2, PRED_HORIZON)
            # project_plot_ellipsoid_2d(ellipsoids_Ab_dict, 1, 3, PRED_HORIZON)
            # project_plot_ellipsoid_2d(ellipsoids_Ab_dict, 3, 5, PRED_HORIZON)
            x1, x2, y1, y2 = [], [], [], []
            for timestep in range(PRED_HORIZON + 1):
                x1.append(pred_states_ego_in_other_frame[timestep][0])
                y1.append(pred_states_ego_in_other_frame[timestep][1])
            plt.scatter(x1, y1, color="blue", marker="o", label="Ego")
            plt.scatter([0], [0], color="red", marker="s", label="Other")
            """
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")  # keep aspect ratio square
            plt.show()

            x1b, y1b, x2b, y2b =  [-state_ego["p_x"], -state_ego["p_x"]+np.cos(-state_ego['psi']+np.pi)], [state_ego['p_y'], state_ego['p_y']+np.sin(-state_ego['psi']+np.pi)], [-state_other["p_x"], -state_other["p_x"]+np.cos(-state_other['psi']+np.pi)], [state_other["p_y"], state_other["p_y"]+np.sin(-state_other['psi']+np.pi)]
            for timestep in range(PRED_HORIZON + 1):
                x1.append(pred_states_ego_in_other_frame[timestep][0])
                y1.append(pred_states_ego_in_other_frame[timestep][1])
            plt.scatter(x1b, y1b, color="blue", marker="o", label="Ego in World")
            plt.scatter(x2b, y2b, color="red", marker="s", label="Other in world")
            """
            x_vals = np.linspace(0, 20, 200)  # range for x
            """
            a, b = np.array([-1.7267251790421878, 1.8078772513812225]), -4.835673698084119 #left
            a2, b2 = np.array([1.0042604722385482,-2.2894237056296953]), -2.1043084288841305# right
            a3, b3 =np.array( [-1.0042604722385482, 2.2894237056296953]), 2.1043084288841305
            a4, b4 = np.array([-2.452516971316602, -0.48493350616764186]), -18.992462222501274
            if a[1] != 0:  # non-vertical line
                y_vals = (b - a[0] * x_vals) / a[1]
                plt.plot(x_vals, y_vals, label=f"front left ")
                y_vals2 = (b2 - a2[0] * x_vals) / a2[1]
                plt.plot(x_vals, y_vals2, label=f"front right ")
                y_vals3 = (b3 - a3[0] * x_vals) / a3[1]
                plt.plot(x_vals, y_vals3, label=f"right left ")
                y_vals4 = (b4 - a4[0] * x_vals) / a4[1]
                plt.plot(x_vals, y_vals4, label=f"right right ")
            else:  # vertical line
                x_line = np.full_like(x_vals, b / a[0])
                plt.plot(x_line, x_vals, label=f"x = {b / a[0]}")

            """

            # xlo = [-10, -10]
            # xhi = [10, 10]
            # xc = np.array([0, 0, 0, 0])
            #
            # xmin = xlo[0] - 0.45 * (xhi[0] - xlo[0])
            # xmax = xhi[0] + 0.45 * (xhi[0] - xlo[0])
            # ymin = xlo[1] - 0.45 * (xhi[1] - xlo[1])
            # ymax = xhi[1] + 0.45 * (xhi[1] - xlo[1])
            #
            # nmgrid = 100
            # x_grid = np.linspace(xmin, xmax, nmgrid)
            # y_grid = np.linspace(ymin, ymax, nmgrid)
            #
            # [X, Y] = np.meshgrid(x_grid, y_grid)
            # Zxy_full = np.zeros((nmgrid, nmgrid))
            # for i in range(nmgrid):
            #     for j in range(nmgrid):
            #         Zxy_full[i, j] = proj_to_2_plotting(X[i, j], Y[i, j], ellipsoids_Ab_dict[2][0], ellipsoids_Ab_dict[2][1], 2, xc)

            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")  # keep aspect ratio square
            plt.show()

        # select correct rechable sets

        # compute atomic predicate intervals
        start = time.perf_counter()
        atomic_interval_dict = get_atomic_intervals(pred_states_ego_in_other_frame, ellipsoids_Ab_dict,
                                                    robustness_fun_dict)
        end = time.perf_counter()
        print(f"Elapsed: {end - start:.6f} sec")

        # compute specification intervals for persistent encounter
        robustness_head = get_ISTL_persistent_encounter(atomic_interval_dict)

        # waypoint generation in tank cosy directly
        if (robustness_head.high > 0) and not maneuvering:
            waypoints_man = get_wp(state_ego)
            waypoints_man.extend(waypoints)
            waypoints = waypoints_man
            maneuvering = True

        # logging
        logging_dict[i] = {"step": i, "ego": copy.deepcopy(state_ego), "other": copy.deepcopy(state_other)}
        i += 1
        # print(state_ego)
        # print(state_other)
        # print(i)

        done, waypoints, ros_dict = step_mock_sim(waypoints, ros_dict)

    return logging_dict


if __name__ == '__main__':
    main(ros_dict)