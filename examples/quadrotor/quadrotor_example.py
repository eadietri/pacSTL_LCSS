from quadrotor_utils import HeightLater, HeightAlways, VelocityBound, AngularVelocityBound, make_quadrotor_samples
from pacSTL.pacSTL_utils import SignalTemporalLogic, EllipsoidalSignalTemporalLogic
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

    HL = HeightLater()
    HL_prime = HeightLater(threshold=0.85)
    HA = HeightAlways()
    VB = VelocityBound()
    AVB = AngularVelocityBound()

    # 1. get atomic predicates
    atomic_interval_dict = {}
    atomic_interval_dict['HeightLater'] ={}
    atomic_interval_dict['HeightLater_prime'] = {}
    atomic_interval_dict['HeightAlways'] = {}
    atomic_interval_dict['VelocityBounds'] = {}

    i = 1
    while i <= PRED_HORIZON:

        atomic_interval_dict['HeightLater'][i] = HL.evaluate_for_states(states[i], i)
        atomic_interval_dict['HeightLater_prime'][i] = HL_prime.evaluate_for_states(states[i], i)
        atomic_interval_dict['HeightAlways'][i] = HA.evaluate_for_states(
            states[i], i)
        atomic_interval_dict['VelocityBounds'][i] = SignalTemporalLogic.disjunction({0: VB.evaluate_for_states(states[i], i), 1: AVB.evaluate_for_states(states[i], i)})

        i += 1

    # specification evaluation
    full_time_horizon = list(range(1, PRED_HORIZON+1))
    sub_spec_1 = SignalTemporalLogic.eventually_globally(atomic_interval_dict['HeightLater'], [1,2,3, 4], None)
    sub_spec_1_prime = SignalTemporalLogic.eventually_globally(atomic_interval_dict['HeightLater_prime'], [1,2,3, 4],
                                                         None)
    sub_spec_2 = SignalTemporalLogic.globally(atomic_interval_dict['HeightAlways'], full_time_horizon)
    sub_spec_3 = SignalTemporalLogic.globally(atomic_interval_dict['VelocityBounds'], full_time_horizon)


    spec_1 = SignalTemporalLogic.conjunction({0: sub_spec_1, 1: sub_spec_2, 2: sub_spec_3})
    spec_2 = SignalTemporalLogic.conjunction({0: sub_spec_1_prime, 1: sub_spec_2, 2: sub_spec_3})

    return spec_1, spec_2, atomic_interval_dict


def eval_with_sets(sets_Ab_dict):


    HL = HeightLater()
    HL_prime = HeightLater(threshold=0.85)
    HA = HeightAlways()
    VB = VelocityBound()
    AVB = AngularVelocityBound()

    atomic_interval_dict = {}
    atomic_interval_dict['HeightLater'] = {}
    atomic_interval_dict['HeightLater_prime'] = {}
    atomic_interval_dict['HeightAlways'] = {}
    atomic_interval_dict['VelocityBounds'] = {}

    i = 1
    start = time.perf_counter()
    while i <= PRED_HORIZON:
        atomic_interval_dict['HeightLater'][i] = HL.compute_robustness(sets_Ab_dict[i][0],
                                                                                sets_Ab_dict[i][1],
                                                                                sets_Ab_dict[i][2], i)
        atomic_interval_dict['HeightLater_prime'][i] = HL_prime.compute_robustness(sets_Ab_dict[i][0],
                                                                                sets_Ab_dict[i][1],
                                                                                sets_Ab_dict[i][2], i)
        atomic_interval_dict['HeightAlways'][i] = HA.compute_robustness(sets_Ab_dict[i][0],
                                                                                sets_Ab_dict[i][1],
                                                                                sets_Ab_dict[i][2], i)
        atomic_interval_dict['VelocityBounds'][i] = EllipsoidalSignalTemporalLogic.disjunction({0: VB.compute_robustness(sets_Ab_dict[i][0],
                                                                                sets_Ab_dict[i][1],
                                                                                sets_Ab_dict[i][2], i), 1: AVB.compute_robustness(sets_Ab_dict[i][0],
                                                                                sets_Ab_dict[i][1],
                                                                                sets_Ab_dict[i][2], i)})

        i += 1

    # specification evaluation
    full_time_horizon = list(range(1, PRED_HORIZON+1))
    # sub_spec_1 = EllipsoidalSignalTemporalLogic.eventually_globally(atomic_interval_dict['HeightLater'], [1,2,3, 4], None)
    sub_spec_1_prime = EllipsoidalSignalTemporalLogic.eventually_globally(atomic_interval_dict['HeightLater_prime'], [1,2,3, 4],
                                                         None)
    sub_spec_2 = EllipsoidalSignalTemporalLogic.globally(atomic_interval_dict['HeightAlways'], full_time_horizon)
    sub_spec_3 = EllipsoidalSignalTemporalLogic.globally(atomic_interval_dict['VelocityBounds'], full_time_horizon)


    #spec_1 = EllipsoidalSignalTemporalLogic.conjunction({0: sub_spec_1, 1: sub_spec_2, 2: sub_spec_3})
    spec_2 = EllipsoidalSignalTemporalLogic.conjunction({0: sub_spec_1_prime, 1: sub_spec_2, 2: sub_spec_3})

    end = time.perf_counter()
    runtime = end - start

    return None, spec_2, atomic_interval_dict, runtime



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
                _, spec_2, atomic_interval_dict, runtime = eval_with_sets(ellipsoids_Ab_dict)
                #print("Robustness spec 1 interval:", spec_1.low, spec_1.high)
                #print("Robustness spec 1 critical time steps:", spec_1.t_low, spec_1.t_high)
                print("Robustness spec 2 interval:", spec_2.low, spec_2.high)
                print("Robustness spec 2 critical time steps:", spec_2.t_low, spec_2.t_high)
                runtimes.append(runtime)

            average_runtime = np.mean(runtimes)
            std_runtime = np.std(runtimes)
            print("Average runtime:", average_runtime)
            print("Std runtime:", std_runtime)


            data_runs[EVAL_TYPE] = {"spec2 interval": [spec_2.low, spec_2.high, [spec_2.t_low, spec_2.t_high], [average_runtime, std_runtime]],
                                }

        elif EVAL_TYPE == "scenario_opt":
            # 3. Create N and M samples --> do we want to use the exact same for the reachable sets?? --- not for now
            ndata = 1500
            data = make_quadrotor_samples(ndata)
            # Evaluate robustness for all of the training trajectories
            spec_1_robustnesses = []
            spec_2_robustnesses = []

            for sample in data:
                states = {
                    time_step: sample[time_step]
                    for time_step in range(1, data.shape[1])
                }

                spec_1, spec_2, atomic_interval_dict = evaluate_traces(states)
                spec_1_robustnesses.append(spec_1.phi)
                spec_2_robustnesses.append(spec_2.phi)

            # 5. Compute spec bounds (min/max) and probabilistic guarantees (bin tail inversion)
            min_h1, max_h1 = np.min(spec_1_robustnesses), np.max(spec_1_robustnesses)
            min_h2, max_h2 = np.min(spec_2_robustnesses), np.max(spec_2_robustnesses)

            print("Robustness spec1 interval:", min_h1, max_h1)
            print("Robustness spec2 interval:", min_h2, max_h2)


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

                spec_1, spec_2, atomic_interval_dict = evaluate_traces(states)

                if spec_1.phi < min_h1 or spec_1.phi > max_h1:
                    spec_1_violations += 1
                if spec_2.phi < min_h2 or spec_2.phi > max_h2:
                    spec_2_violations += 1

            print("Number of spec 1 violations:", spec_1_violations)
            print("Number of spec 2 violations:",spec_2_violations)



            epsilon_h1 = binomial_tail(spec_1_violations, ndata)
            epsilon_h2 = binomial_tail(spec_2_violations, ndata)
            print("Epsilon spec 1:", epsilon_h1)
            print("Epsilon spec 2:", epsilon_h2)

            data_runs[EVAL_TYPE] = { "spec1 interval": [min_h1, max_h1, epsilon_h1],
                        "spec2 interval": [min_h2, max_h2, epsilon_h2]}
