import copy
import pickle
import numpy as np
from examples.vessel_navigation.vessel_navigation_example import get_ISTL_persistent_encounter, get_atomic_intervals, pre_script
from examples.vessel_navigation.vessel_navigation_example import DrillshipSimulator, step_sim, PRED_HORIZON
from pacSTL.pacSTL_utils import PACSignalTemporalLogic, SignalTemporalLogic

def evaluate_from_initial_conditions(initial_conditions_t1, pkl_path, ellipsoids_Ab_dict, robustness_fun_dict):

    with open(pkl_path, 'rb') as f:
        saved_runs = pickle.load(f)

    saved_log = saved_runs[0]['log']
    results = {}

    for ic_idx, init in enumerate(initial_conditions_t1):
        print(f"\n=== Initial Condition {ic_idx + 1} (from step {init['step']}) ===")

        source_step = init['step']
        if source_step not in saved_log:
            print(f"  Step {source_step} not found in saved log, skipping.")
            continue

        pred_states_ego_in_other_frame = saved_log[source_step]['ego_pred']

        simulator = DrillshipSimulator()
        ros_dict_temp = copy.deepcopy(init)

        tau_cmd = np.array(
            [np.random.uniform(0.7, 1.2), np.random.uniform(-0.1, 0.1), 0, 0, 0, np.random.uniform(-0.1, 0.1)],
            dtype=float).reshape(-1, 1)

        # --- 1. roll out other vessel for all PRED_HORIZON steps ---
        drillship_position = {}
        for k in range(1, PRED_HORIZON + 1):
            if k % 5 == 0:
                tau_cmd = np.array(
                    [np.random.uniform(0.7, 1.2), np.random.uniform(-0.1, 0.1), 0, 0, 0, np.random.uniform(-0.1, 0.1)],
                    dtype=float).reshape(-1, 1)

            done, _, ros_dict_temp = step_sim(
                [np.array([ros_dict_temp["other"]["p_x"], ros_dict_temp["other"]["p_y"]])],
                ros_dict_temp, simulator, tau_cmd)

            drillship_position[k] = np.array([
                ros_dict_temp["other"]["p_x"],
                ros_dict_temp["other"]["p_y"],
                ros_dict_temp["other"]["psi"],
                ros_dict_temp["other"]["v_x"],
                ros_dict_temp["other"]["v_y"],
            ])

        # --- 2. pacSTL interval over full spec ---
        atomic_interval_dict = get_atomic_intervals(
            pred_states_ego_in_other_frame, robustness_fun_dict, ellipsoids_Ab_dict)
        robustness_head = get_ISTL_persistent_encounter(atomic_interval_dict)

        # --- 3. point robustness over full spec using rollout ---
        atomic_point_dict = get_atomic_intervals(
            pred_states_ego_in_other_frame, robustness_fun_dict,
            drillship_position=drillship_position, pacstl=False)
        phi_head = get_ISTL_persistent_encounter(atomic_point_dict, pacstl=False)

        # --- 4. check ---
        within = robustness_head.low <= phi_head.phi <= robustness_head.high
        t_low  = robustness_head.t_low
        t_high = robustness_head.t_high
        violation_at_characteristic = (not within) and ((phi_head.t_phi == t_low) or (phi_head.t_phi == t_high))

        print(f"  interval=[{robustness_head.low:.4f}, {robustness_head.high:.4f}], "
            f"phi={phi_head.phi:.4f}, t_phi={phi_head.t_phi} "
            f"-> {'WITHIN' if within else 'VIOLATION'}"
            + (f" ** AT CHARACTERISTIC STEP" if violation_at_characteristic else
                f" not at characteristic (t_low={t_low}, t_high={t_high})" if not within else ""))

        results[ic_idx] = {
            "init":                      init,
            "source_step":               source_step,
            "interval_low":              robustness_head.low,
            "interval_high":             robustness_head.high,
            "phi":                       phi_head.phi,
            "t_phi":                     phi_head.t_phi,
            "t_low":                     t_low,
            "t_high":                    t_high,
            "within":                    within,
            "violation_at_characteristic": violation_at_characteristic
        }

    return results

def run_violation_study(initial_conditions_t1, pkl_path, ellipsoids_Ab_dict, robustness_fun_dict, n_runs=1500):

    with open(pkl_path, 'rb') as f:
        saved_runs = pickle.load(f)
    saved_log = saved_runs[0]['log']

    # per IC counters
    study_results = {ic_idx: {
        "total": 0,
        "violations": 0,
        "violations_at_characteristic": 0,
        "violations_not_at_characteristic": 0,
    } for ic_idx in range(len(initial_conditions_t1))}

    for run in range(n_runs):
        if run % 100 == 0:
            print(f"  Run {run}/{n_runs}...")

        for ic_idx, init in enumerate(initial_conditions_t1):
            source_step = init['step']
            pred_states_ego_in_other_frame = saved_log[source_step]['ego_pred']

            simulator = DrillshipSimulator()
            ros_dict_temp = copy.deepcopy(init)
            tau_cmd = np.array(
                [np.random.uniform(0.7, 1.2), np.random.uniform(-0.1, 0.1), 0, 0, 0, np.random.uniform(-0.1, 0.1)],
                dtype=float).reshape(-1, 1)

            # roll out other vessel
            drillship_position = {}
            for k in range(1, PRED_HORIZON + 1):
                if k % 5 == 0:
                    tau_cmd = np.array(
                        [np.random.uniform(0.7, 1.2), np.random.uniform(-0.1, 0.1), 0, 0, 0, np.random.uniform(-0.1, 0.1)],
                        dtype=float).reshape(-1, 1)

                done, _, ros_dict_temp = step_sim(
                    [np.array([ros_dict_temp["other"]["p_x"], ros_dict_temp["other"]["p_y"]])],
                    ros_dict_temp, simulator, tau_cmd)

                drillship_position[k] = np.array([
                    ros_dict_temp["other"]["p_x"],
                    ros_dict_temp["other"]["p_y"],
                    ros_dict_temp["other"]["psi"],
                    ros_dict_temp["other"]["v_x"],
                    ros_dict_temp["other"]["v_y"],
                ])

            # pacSTL interval
            atomic_interval_dict = get_atomic_intervals(
                pred_states_ego_in_other_frame, robustness_fun_dict, ellipsoids_Ab_dict)
            robustness_head = get_ISTL_persistent_encounter(atomic_interval_dict)

            # point robustness
            atomic_point_dict = get_atomic_intervals(
                pred_states_ego_in_other_frame, robustness_fun_dict,
                drillship_position=drillship_position, pacstl=False)
            phi_head = get_ISTL_persistent_encounter(atomic_point_dict, pacstl=False)

            # check
            within = robustness_head.low <= phi_head.phi <= robustness_head.high
            t_low  = robustness_head.t_low
            t_high = robustness_head.t_high
            violation_at_characteristic = (not within) and ((phi_head.t_phi == t_low) or (phi_head.t_phi == t_high))

            study_results[ic_idx]["total"] += 1
            if not within:
                study_results[ic_idx]["violations"] += 1
                if violation_at_characteristic:
                    study_results[ic_idx]["violations_at_characteristic"] += 1
                else:
                    study_results[ic_idx]["violations_not_at_characteristic"] += 1

    # summary
    print(f"\n{'='*50}")
    print(f"VIOLATION STUDY SUMMARY ({n_runs} runs)")
    print(f"{'='*50}")
    for ic_idx, counts in study_results.items():
        total = counts["total"]
        viol  = counts["violations"]
        char  = counts["violations_at_characteristic"]
        not_char = counts["violations_not_at_characteristic"]
        print(f"\n  IC {ic_idx + 1} (source_step={initial_conditions_t1[ic_idx]['step']}):")
        print(f"    Total runs:                      {total}")
        print(f"    Violations:                      {viol}/{total} ({100*viol/total:.1f}%)")
        print(f"    At characteristic timestep:      {char}/{viol if viol > 0 else 1} ({100*char/viol:.1f}%)" if viol > 0 else f"    At characteristic timestep:      0")
        print(f"    NOT at characteristic timestep:  {not_char}/{viol if viol > 0 else 1} ({100*not_char/viol:.1f}%)" if viol > 0 else f"    NOT at characteristic timestep:  0")

    return study_results


if __name__ == "__main__":
    pkl_path = "examples/vessel_navigation/vessel_multi_runs.pkl"

    ellipsoids_Ab_dict, robustness_fun_dict = pre_script()
    with open(pkl_path, 'rb') as f:
        saved_runs = pickle.load(f)
    saved_log = saved_runs[0]['log']

    # build ICs from logged steps directly
    def make_ic(step):
        entry = saved_log[step]
        return {
            "ego": dict(entry['ego']),
            "other": dict(entry['other']),
            "initial_other": {
                "p_x": entry['other']['p_x'],
                "p_y": entry['other']['p_y'],
                "psi": entry['other']['psi'],
            },
            "step": step
        }

    initial_conditions_t1 = [make_ic(s) for s in [1, 2, 3]]
    initial_conditions_t2 = [make_ic(s) for s in [25, 26, 27]]
    initial_conditions_t5 = [make_ic(s) for s in [35, 85, 105]]

    for label, ics in [("t1", initial_conditions_t1), 
                       ("t2", initial_conditions_t2), 
                       ("t5", initial_conditions_t5)]:
        print(f"\n{'='*50}")
        print(f"VIOLATION STUDY — {label}")
        print(f"{'='*50}")
        run_violation_study(ics, pkl_path, ellipsoids_Ab_dict, robustness_fun_dict, n_runs=1500)

    # results = evaluate_from_initial_conditions(
    #     initial_conditions_t1, pkl_path, ellipsoids_Ab_dict, robustness_fun_dict)

    # # Print summary of results
    # for ic_idx, res in results.items():
    #     print(f"\n=== Summary for Initial Condition {ic_idx + 1} (from step {res['source_step']}) ===")
    #     within = res['within']
    #     violation_at_characteristic = res['violation_at_characteristic']
    #     violations = [] if within else [(ic_idx, res)]
    #     char_violations = [] if not violation_at_characteristic else [(ic_idx, res)]
    #     print(f"  Violations: {len(violations)}")
    #     if not within:
    #         print(f"    phi={res['phi']:.4f} outside [{res['interval_low']:.4f}, {res['interval_high']:.4f}]"
    #             + (" ** CHARACTERISTIC" if violation_at_characteristic else ""))