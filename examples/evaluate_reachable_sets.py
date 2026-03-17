import numpy as np
import os
import pickle
from reachability_utils.binomial import calculate_epsilon_ellipsoid, calculate_epsilon_misses, calculate_epsilon_tube_ellipsoid
from examples.vessel_navigation.vessel_utils import make_vessel_samples
from examples.quadrotor.quadrotor_utils import make_quadrotor_samples

import matplotlib.pyplot as plt

# Vessel:
# epsilon for vessel reachable tube:  0.1069038165032687
# epsilon for vessel ellipsoid t=1:  0.045339298531341304
# epsilon for vessel ellipsoid t=2:  0.044308993920390914
# epsilon for vessel ellipsoid t=3:  0.045339298531341304
# epsilon for vessel ellipsoid t=4:  0.050388731511159586
# epsilon for vessel ellipsoid t=5:  0.034621301651395646

# Quadrotor:
# epsilon for quadrotor reachable tube:  0.47261742379593386
# epsilon for quadrotor ellipsoid t=0.25:  0.06940041729725141
# epsilon for quadrotor ellipsoid t=0.5:  0.060087856017458335
# epsilon for quadrotor ellipsoid t=0.75:  0.06385222153278249
# epsilon for quadrotor ellipsoid t=1:  0.07122729940609279
# epsilon for quadrotor ellipsoid t=1.25:  0.045339298531341304
# epsilon for quadrotor ellipsoid t=1.5:  0.06103430682071637
# epsilon for quadrotor ellipsoid t=1.75:  0.0543242523902271
# epsilon for quadrotor ellipsoid t=2:  0.05818334449836276
# epsilon for quadrotor ellipsoid t=2.25:  0.060087856017458335
# epsilon for quadrotor ellipsoid t=2.5:  0.05818334449836276
# epsilon for quadrotor ellipsoid t=2.75:  0.0684835505946733
# epsilon for quadrotor ellipsoid t=3:  0.056262233172135113
# epsilon for quadrotor ellipsoid t=3.25:  0.06756335372954136
# epsilon for quadrotor ellipsoid t=3.5:  0.05334803306883369
# epsilon for quadrotor ellipsoid t=3.75:  0.0543242523902271
# epsilon for quadrotor ellipsoid t=4:  0.06940041729725141
# epsilon for quadrotor ellipsoid t=4.25:  0.07304294091556593
# epsilon for quadrotor ellipsoid t=4.5:  0.07394733179009132
# epsilon for quadrotor ellipsoid t=4.75:  0.06291630268217446
    

def get_reachable_sets(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data

def vessel_traj_misses(data, sets, ax, ndata):
    misses = 0
    all_steps = []

    step = []
    for time_step in range(1, data.shape[1]):
        for i in range(ndata):
            for j in range(data.shape[2]):
                step.append(data[i, time_step, j])
        all_steps.append(np.array(step))
        step = []
    
    for t in range(0, data.shape[1]-1):
        misses += calculate_epsilon_misses(all_steps[t], sets[t+1][0], sets[t+1][1], ndata)

    opt = calculate_epsilon_tube_ellipsoid(misses, ndata)
    return opt

def quadrotor_traj_misses(data, sets, ax, ndata):
    misses = 0
    all_steps = []

    step = []
    for time_step in range(1, data.shape[1]):
        for i in range(ndata):
            step.append(data[i, time_step])
        all_steps.append(np.array(step))
        step = []
    
    # for t in range(1, data.shape[1]):
        misses += calculate_epsilon_misses(all_steps[time_step-1], sets[time_step][0], sets[time_step][1], ndata)
    
    print("total misses: ", misses)
    opt = calculate_epsilon_tube_ellipsoid(misses, ndata)

    return opt

def quadrotor_time_point_set(Aq, bq, t1, t2):
    ndata = 1500
    quadrotor_test = make_quadrotor_samples(ndata)
    qstep = []
    for time_step in range(t1, t2):
        for i in range(ndata):
            qstep.append(quadrotor_test[i, time_step])

    qopt = calculate_epsilon_ellipsoid(qstep, Aq, bq, ndata)
    return qopt

def vessel_time_point_set(A, b, t1, t2):
    ndata = 1500
    test_set = make_vessel_samples(ndata)
    step = []
    for time_step in range(t1, t2):
        for i in range(ndata):
            for j in range(test_set.shape[2]):
                step.append(test_set[i, time_step, j])
    opt1 = calculate_epsilon_ellipsoid(step, A, b, ndata)
    return opt1

if __name__ == '__main__':
    fig, ax = plt.subplots()

    ### Evaluate Vessel reachable tube + sets:

    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    file = os.path.join(script_directory, 'reachable_sets_vessel.pkl')

    vessel_sets = get_reachable_sets(file)

    ndata = 1500
    data_test_tube = make_vessel_samples(ndata)
    opt_vessel = vessel_traj_misses(data_test_tube, vessel_sets, ax, ndata)
    print("epsilon for vessel reachable tube: ", opt_vessel) 

    # reach set t=1
    A1, b1, c1 = vessel_sets[1]

    # reach set t=2
    A2, b2, c2 = vessel_sets[2]

    # reach set t=3
    A3, b3, c3 = vessel_sets[3]

    # reach set t=4
    A4, b4, c4 = vessel_sets[4]

    # reach set t=5
    A5, b5, c5 = vessel_sets[5]

    # # evaluate set at t=1
    opt1 = vessel_time_point_set(A1, b1, 1, 2)
    print("epsilon for vessel ellipsoid t=1: ", opt1)

    # # evaluate set at t=2
    opt2 = vessel_time_point_set(A2, b2, 2, 3)
    print("epsilon for vessel ellipsoid t=2: ", opt2)

    # # evaluate set at t=3
    opt3 = vessel_time_point_set(A3, b3, 3, 4)
    print("epsilon for vessel ellipsoid t=3: ", opt3)

    # # evaluate set at t=4
    opt4 = vessel_time_point_set(A4, b4, 4, 5)
    print("epsilon for vessel ellipsoid t=4: ", opt4)

    # # evaluate set at t=5
    opt5 = vessel_time_point_set(A5, b5, 5, 6)
    print("epsilon for vessel ellipsoid t=5: ", opt5)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

### Evaluate Quadrotor reachable tube + sets:

    file2 = os.path.join(script_directory, 'reachable_sets_quadrotor.pkl')

    quadrotor_sets = get_reachable_sets(file2)

    ndata = 1500
    quadrotor_test_tube = make_quadrotor_samples(ndata)
    opt_quad = quadrotor_traj_misses(quadrotor_test_tube, quadrotor_sets, ax, ndata)
    print("epsilon for quadrotor reachable tube: ", opt_quad) 

    # reach set t=0.25
    Aq1, bq1, cq1 = quadrotor_sets[1]

    # reach set t=0.5
    Aq2, bq2, cq2 = quadrotor_sets[2]

    # reach set t=0.75
    Aq3, bq3, cq3 = quadrotor_sets[3]

    # reach set t=1
    Aq4, bq4, cq4 = quadrotor_sets[4]

    # reach set t=1.25
    Aq5, bq5, cq5 = quadrotor_sets[5]

    # reach set t=1.5
    Aq6, bq6, cq6 = quadrotor_sets[6]

    # reach set t=1.75
    Aq7, bq7, cq7 = quadrotor_sets[7]

    # reach set t=2
    Aq8, bq8, cq8 = quadrotor_sets[8]

    # reach set t=2.25
    Aq9, bq9, cq9 = quadrotor_sets[9]

    # reach set t=2.5
    Aq10, bq10, cq10 = quadrotor_sets[10]

    # reach set t=2.75
    Aq11, bq11, cq11 = quadrotor_sets[11]

    # reach set t=3
    Aq12, bq12, cq12 = quadrotor_sets[12]

    # reach set t=3.25
    Aq13, bq13, cq13 = quadrotor_sets[13]

    # reach set t=3.5
    Aq14, bq14, cq14 = quadrotor_sets[14]

    # reach set t=3.75
    Aq15, bq15, cq15 = quadrotor_sets[15]

    # reach set t=4
    Aq16, bq16, cq16 = quadrotor_sets[16]

    # reach set t=4.25
    Aq17, bq17, cq17 = quadrotor_sets[17]

    # reach set t=4.5
    Aq18, bq18, cq18 = quadrotor_sets[18]

    # reach set t=4.75
    Aq19, bq19, cq19 = quadrotor_sets[19]

    # # evaluate set at t=0.25
    qopt1 = quadrotor_time_point_set(Aq1, bq1, 1, 2)
    print("epsilon for quadrotor ellipsoid t=0.25: ", qopt1)

    # # evaluate set at t=0.5
    qopt2 = quadrotor_time_point_set(Aq2, bq2, 2, 3)
    print("epsilon for quadrotor ellipsoid t=0.5: ", qopt2)

    # # evaluate set at t=0.75
    qopt3 = quadrotor_time_point_set(Aq3, bq3, 3, 4)
    print("epsilon for quadrotor ellipsoid t=0.75: ", qopt3)

    # # evaluate set at t=1
    qopt4 = quadrotor_time_point_set(Aq4, bq4, 4, 5)
    print("epsilon for quadrotor ellipsoid t=1: ", qopt4)

    # # evaluate set at t=1.25
    qopt5 = quadrotor_time_point_set(Aq5, bq5, 5, 6)
    print("epsilon for quadrotor ellipsoid t=1.25: ", qopt5)

    # # evaluate set at t=1.5
    qopt6 = quadrotor_time_point_set(Aq6, bq6, 6, 7)
    print("epsilon for quadrotor ellipsoid t=1.5: ", qopt6)

    # # evaluate set at t=1.75

    qopt7 = quadrotor_time_point_set(Aq7, bq7, 7, 8)
    print("epsilon for quadrotor ellipsoid t=1.75: ", qopt7)

    # # evaluate set at t=2
    qopt8 = quadrotor_time_point_set(Aq8, bq8, 8, 9)
    print("epsilon for quadrotor ellipsoid t=2: ", qopt8)

    # # evaluate set at t=2.25
    qopt9 = quadrotor_time_point_set(Aq9, bq9, 9, 10)
    print("epsilon for quadrotor ellipsoid t=2.25: ", qopt9)

    # # evaluate set at t=2.5
    qopt10 = quadrotor_time_point_set(Aq10, bq10, 10, 11)
    print("epsilon for quadrotor ellipsoid t=2.5: ", qopt10)

    # # evaluate set at t=2.75
    qopt11 = quadrotor_time_point_set(Aq11, bq11, 11, 12)
    print("epsilon for quadrotor ellipsoid t=2.75: ", qopt11)

    # # evaluate set at t=3
    qopt12 = quadrotor_time_point_set(Aq12, bq12, 12, 13)
    print("epsilon for quadrotor ellipsoid t=3: ", qopt12)

    # # evaluate set at t=3.25
    qopt13 = quadrotor_time_point_set(Aq13, bq13, 13, 14)
    print("epsilon for quadrotor ellipsoid t=3.25: ", qopt13)

    # # evaluate set at t=3.5
    qopt14 = quadrotor_time_point_set(Aq14, bq14, 14, 15)
    print("epsilon for quadrotor ellipsoid t=3.5: ", qopt14)

    # # evaluate set at t=3.75
    qopt15 = quadrotor_time_point_set(Aq15, bq15, 15, 16)
    print("epsilon for quadrotor ellipsoid t=3.75: ", qopt15)

    # # evaluate set at t=4
    qopt16 = quadrotor_time_point_set(Aq16, bq16, 16, 17)
    print("epsilon for quadrotor ellipsoid t=4: ", qopt16)

    # # evaluate set at t=4.25
    qopt17 = quadrotor_time_point_set(Aq17, bq17, 17, 18)
    print("epsilon for quadrotor ellipsoid t=4.25: ", qopt17)

    # # evaluate set at t=4.5
    qopt18 = quadrotor_time_point_set(Aq18, bq18, 18, 19)
    print("epsilon for quadrotor ellipsoid t=4.5: ", qopt18)

    # # evaluate set at t=4.75
    qopt19 = quadrotor_time_point_set(Aq19, bq19, 19, 20)
    print("epsilon for quadrotor ellipsoid t=4.75: ", qopt19)