# make mock-sim version that monitors simpfied spec (infront and collision risk) for head-on case

import numpy as np
import copy
import time
import os
import enum

from pacSTL.pacSTL_utils import EllipsoidalSignalTemporalLogic
from reachability_utils.trigonometry_utils import rotation_matrix
from examples.evaluate_reachable_sets import get_reachable_sets
from vessel_utils import S_DDOT_MAX, S_DDOT_MAX_DRILL, T_H, R_EGO, R_EGO_DRILL, S_DOT_MAX_DRILL, \
    D_DOT_MAX_DRILL, InFrontRobustness, CollisionRobustness
from vessel_utils import step_mock_sim, DrillshipSimulator

import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
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

MAX_TRAJ_LENGTH = 100
PRED_HORIZON = 5  # in DT
DT = 0.5  # in s
MIN_MANEUVER_STEPS = 10
MIN_TURNING_ANGLE = -0.8  # in rad
TIME_TURNING = DT * 45  # in s
TIME_PARALLEL = DT * 30  # in s
DEBUG = False


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


    if CONFIGVESSELS == ConfigType.VD or CONFIGVESSELS == ConfigType.DD:
        ellipsoids_Ab_dict = get_reachable_sets(os.path.join(os.path.dirname(__file__)) + "/reachable_sets_vessel.pkl")
    else:
        print("sets not computed.")
        raise ValueError

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
        print(np.linalg.norm(pred_states_ego_in_other_frame[0][0:2]) - np.linalg.norm(
            [state_other["p_x"] - state_ego["p_x"], state_other["p_y"] - state_ego["p_y"]]))
        print(abs(pred_states_ego_in_other_frame[0][2]) - abs(state_ego['psi'] - state_other['psi']))

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
            # random initial tau
            # forces x, y, z, moments x, y, z
            tau_cmd = np.array(
                [np.random.uniform(0.7, 1.2), np.random.uniform(-0.1, 0.1), 0, 0, 0, np.random.uniform(-0.1, 0.1)],
                dtype=float).reshape(-1, 1)

        done, waypoints, ros_dict = step_mock_sim(waypoints, ros_dict, simulator,tau_cmd)

        i += 1

    return logging_dict

def evaluation_of_runs(data):

    # --- 1. Initialize Data Containers ---
    times = []
    rob_lower, rob_upper = [], []
    ego_px, ego_py = [], []
    other_px, other_py = [], []

    # Containers for the encounter markers
    enc_times, enc_rob_low, enc_rob_high = [], [], []
    enc_ego_px, enc_ego_py = [], []
    enc_other_px, enc_other_py = [], []

    runtimes = []
    rob_integers = []

    dt = 0.5

    # --- 2. Extract Data ---
    # Sorting keys ensures the time steps are processed in chronological order
    for step in sorted(data.keys()):
        entry = data[step]

        # Calculate time
        t = step * dt
        times.append(t)

        # Robustness bounds (first 2 entries)
        r_low = entry['robustness'][0]
        r_high = entry['robustness'][1]
        rob_lower.append(r_low)
        rob_upper.append(r_high)

        # Trajectories
        e_px, e_py = entry['ego']['p_x'], entry['ego']['p_y']
        o_px, o_py = entry['other']['p_x'], entry['other']['p_y']
        ego_px.append(e_px)
        ego_py.append(e_py)
        other_px.append(o_px)
        other_py.append(o_py)

        # Check for encounter (if it is a list, it's an encounter)
        if isinstance(entry['encounter'], list):
            enc_times.append(t)
            enc_rob_low.append(r_low)
            enc_rob_high.append(r_high)
            enc_ego_px.append(e_px)
            enc_ego_py.append(e_py)
            enc_other_px.append(o_px)
            enc_other_py.append(o_py)

        # Stats collection
        runtimes.append(entry['runtime'])
        # logic according to epsilon values for the sets -- always take the timestep that corresponds to the most conservative epsilon
        if 4 in entry['robustness'][-2:]:
            rob_integers.extend([4])
        elif 1 in entry['robustness'][-2:]:
            rob_integers.extend([1])
        elif 3 in entry['robustness'][-2:]:
            rob_integers.extend([3])
        elif 2 in entry['robustness'][-2:]:
            rob_integers.extend([2])
        elif 5 in entry['robustness'][-2:]:
            rob_integers.extend([5]) 

    # --- 3. Compute and Print Evaluations ---
    print("=== Evaluation Results ===")

    # Runtimes
    avg_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)
    print(f"Runtime -> Average: {avg_runtime:.5f}s, Std Dev: {std_runtime:.5f}s")

    # Robustness Integers Count
    counts = Counter(rob_integers)
    print("\nRobustness Integers Frequency (from the last two values):")
    for i in range(1, 6):
        print(f"  Integer {i}: occurs {counts.get(i, 0)} time(s)")
    print("==========================\n")

    # --- 4. Plotting ---

    # Plot 1: Robustness over Time
    plt.figure(figsize=(10, 5))
    plt.plot(times, rob_lower, label='Lower Robustness', color='blue')
    plt.plot(times, rob_upper, label='Upper Robustness', color='orange')

    # Add Encounter Markers
    plt.scatter(enc_times, enc_rob_low, color='red', marker='x', s=80, zorder=5, label='Encounter (Lower)')
    plt.scatter(enc_times, enc_rob_high, color='darkred', marker='o', s=80, zorder=5, label='Encounter (Upper)')

    plt.title('Robustness Bounds Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Robustness Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Trajectories
    plt.figure(figsize=(8, 8))
    plt.plot(ego_px, ego_py, label='Ego Trajectory', color='blue')
    plt.plot(other_px, other_py, label='Other Trajectory', color='green')

    # Add Encounter Markers
    plt.scatter(enc_ego_px, enc_ego_py, color='blue', marker='x', s=100, zorder=5, label='Ego at Encounter')
    plt.scatter(enc_other_px, enc_other_py, color='green', marker='x', s=100, zorder=5, label='Other at Encounter')

    plt.title('Ego vs Other Trajectories')
    plt.xlabel('Position X ($p_x$)')
    plt.ylabel('Position Y ($p_y$)')
    plt.axis('equal')  # Ensures the spatial scale is 1:1, crucial for trajectory plots
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def export_logs_to_csv(logs_list, dt=0.5, rob_filename="robustness_traces.csv", states_filename="vessel_states.csv"):
    """
    logs_list: A list containing your 3 dictionary logs (e.g., [log1, log2, log3])
    """
    rob_data = []
    states_data = []

    # Assuming all logs share the same time steps, we use the keys from the first log
    time_steps = sorted(logs_list[1].keys())

    for step in time_steps:
        t = step * dt

        # Initialize rows with time
        rob_row = {'time': t}
        state_row = {}

        # Loop through each of the 3 runs
        for i, log in enumerate(logs_list):
            run_idx = i + 1  # 1, 2, 3
            entry = log[step]

            # Robustness (Wide format)
            rob_row[f'run{run_idx}_lower'] = entry['robustness'][0]
            rob_row[f'run{run_idx}_upper'] = entry['robustness'][1]

            # States (Wide format)
            state_row[f'run{run_idx}_ego_px'] = entry['ego']['p_x']
            state_row[f'run{run_idx}_ego_py'] = -entry['ego']['p_y']
            state_row[f'run{run_idx}_ego_psi'] = entry['ego']['psi']

            state_row[f'run{run_idx}_other_px'] = entry['other']['p_x']
            state_row[f'run{run_idx}_other_py'] = -entry['other']['p_y']
            state_row[f'run{run_idx}_other_psi'] = entry['other']['psi']

        rob_data.append(rob_row)
        states_data.append(state_row)

    # Convert to DataFrames and export
    df_rob = pd.DataFrame(rob_data)
    df_states = pd.DataFrame(states_data)

    # Export without the index column so PGFPlots reads it cleanly
    df_rob.to_csv(rob_filename, index=False)
    df_states.to_csv(states_filename, index=False)
    print(f"Exported: {rob_filename} and {states_filename}")

if __name__ == '__main__':

    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)

    logs = main(ros_dict)
    file = os.path.join(script_directory, 'vessel_example_traces.pkl')
    with open(file, 'wb') as f:
        pickle.dump(logs, f)