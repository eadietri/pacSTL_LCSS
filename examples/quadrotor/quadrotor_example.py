from quadrotor_utils import HeightLater, HeightAlways, VelocityBound, AngularVelocityBound, make_quadrotor_samples
from pacSTL.pacSTL_utils import SignalTemporalLogic, PACSignalTemporalLogic
from reachability_utils.binomial import binomial_tail
from examples.evaluate_reachable_sets import get_reachable_sets

import numpy as np
import os
import time


PRED_HORIZON = 19


def evaluate_traces(states):
    """

    Parameters
    ----------
    states - dict for time steps 1 - 20
    Returns
    -------


    """

    HL = HeightLater(threshold=0.85)
    HA = HeightAlways()
    VB = VelocityBound()
    AVB = AngularVelocityBound()

    # 1. get atomic predicates
    atomic_interval_dict = {}
    atomic_interval_dict['HeightLater'] ={}
    atomic_interval_dict['HeightAlways'] = {}
    atomic_interval_dict['VelocityBounds'] = {}

    i = 1
    while i <= PRED_HORIZON:

        atomic_interval_dict['HeightLater'][i] = HL.evaluate_for_states(states[i], i)
        atomic_interval_dict['HeightAlways'][i] = HA.evaluate_for_states(
            states[i], i)
        atomic_interval_dict['VelocityBounds'][i] = SignalTemporalLogic.disjunction({0: VB.evaluate_for_states(states[i], i), 1: AVB.evaluate_for_states(states[i], i)})

        i += 1

    # specification evaluation
    full_time_horizon = list(range(1, PRED_HORIZON+1))
    sub_spec_1 = SignalTemporalLogic.eventually_globally(atomic_interval_dict['HeightLater'], [1,2,3, 4], None)
    sub_spec_2 = SignalTemporalLogic.globally(atomic_interval_dict['HeightAlways'], full_time_horizon)
    sub_spec_3 = SignalTemporalLogic.globally(atomic_interval_dict['VelocityBounds'], full_time_horizon)


    spec_1 = SignalTemporalLogic.conjunction({0: sub_spec_1, 1: sub_spec_2, 2: sub_spec_3})

    return spec_1, atomic_interval_dict


def eval_with_sets(sets_Ab_dict):

    HL = HeightLater(threshold=0.85)
    HA = HeightAlways()
    VB = VelocityBound()
    AVB = AngularVelocityBound()

    atomic_interval_dict = {}
    atomic_interval_dict['HeightLater'] = {}
    atomic_interval_dict['HeightAlways'] = {}
    atomic_interval_dict['VelocityBounds'] = {}

    i = 1
    start = time.perf_counter()
    while i <= PRED_HORIZON:
        atomic_interval_dict['HeightLater'][i] = HL.compute_robustness(sets_Ab_dict[i][0],
                                                                                sets_Ab_dict[i][1],
                                                                                sets_Ab_dict[i][2], i)
        atomic_interval_dict['HeightAlways'][i] = HA.compute_robustness(sets_Ab_dict[i][0],
                                                                                sets_Ab_dict[i][1],
                                                                                sets_Ab_dict[i][2], i)
        atomic_interval_dict['VelocityBounds'][i] = PACSignalTemporalLogic.disjunction({0: VB.compute_robustness(sets_Ab_dict[i][0],
                                                                                                                 sets_Ab_dict[i][1],
                                                                                                                 sets_Ab_dict[i][2], i), 1: AVB.compute_robustness(sets_Ab_dict[i][0],
                                                                                sets_Ab_dict[i][1],
                                                                                sets_Ab_dict[i][2], i)})

        i += 1

    # specification evaluation
    full_time_horizon = list(range(1, PRED_HORIZON+1))
    sub_spec_1 = PACSignalTemporalLogic.eventually_globally(atomic_interval_dict['HeightLater'], [1,2,3, 4], None)
    sub_spec_2 = PACSignalTemporalLogic.globally(atomic_interval_dict['HeightAlways'], full_time_horizon)
    sub_spec_3 = PACSignalTemporalLogic.globally(atomic_interval_dict['VelocityBounds'], full_time_horizon)


    spec_1 = PACSignalTemporalLogic.conjunction({0: sub_spec_1, 1: sub_spec_2, 2: sub_spec_3})

    end = time.perf_counter()
    runtime = end - start

    return spec_1, atomic_interval_dict, runtime



if __name__ == '__main__':

    data_runs = {}
    for EVAL_TYPE in ["ellipsoid", "scenario_opt"]: 

        if EVAL_TYPE == "ellipsoid":

            # 1. Load reachable sets
            ellipsoids_Ab_dict = get_reachable_sets(
                os.path.join(os.path.dirname(__file__)) + "/reachable_sets_quadrotor.pkl")

            runtimes = []

            for i in range(10): #loop for runtime eval --- set to 1 otherwise

                # 2. Compute pacSTL and store also intermediate results
                spec_1 , atomic_interval_dict, runtime = eval_with_sets(ellipsoids_Ab_dict)
                print("Robustness spec 1 interval:", spec_1.low, spec_1.high)
                print("Robustness spec 1 critical time steps:", spec_1.t_low, spec_1.t_high)
                runtimes.append(runtime)

            average_runtime = np.mean(runtimes)
            std_runtime = np.std(runtimes)
            print("Average runtime:", average_runtime)
            print("Std runtime:", std_runtime)


        elif EVAL_TYPE == "scenario_opt":
            ndata = 1500
            data = make_quadrotor_samples(ndata)
            # Evaluate robustness for all of the training trajectories
            spec_1_robustnesses = []

            for sample in data:
                states = {
                    time_step: sample[time_step]
                    for time_step in range(1, data.shape[1])
                }

                spec_1, atomic_interval_dict = evaluate_traces(states)
                spec_1_robustnesses.append(spec_1.phi)

            # 5. Compute spec bounds (min/max) and probabilistic guarantees (bin tail inversion)
            min_h1, max_h1 = np.min(spec_1_robustnesses), np.max(spec_1_robustnesses)

            print("Robustness spec1 interval:", min_h1, max_h1)


            # get testing sim data
            ndata = 1500
            data = make_quadrotor_samples(ndata)

            # Evaluate robustness for all of the testing trajectories
            spec_1_violations = 0
            spec_2_violations = 0

            for sample in data:
                states = {
                    time_step: sample[time_step]
                    for time_step in range(1, data.shape[1])
                }

                spec_1, atomic_interval_dict = evaluate_traces(states)

                if spec_1.phi < min_h1 or spec_1.phi > max_h1:
                    spec_1_violations += 1

            print("Number of spec 1 violations:", spec_1_violations)

            epsilon_h1 = binomial_tail(spec_1_violations, ndata)
            print("Epsilon spec 1:", epsilon_h1)

