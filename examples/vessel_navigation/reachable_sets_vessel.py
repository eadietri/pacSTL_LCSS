import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from vessel_utils import make_vessel_samples
from reachability_utils.ellipsoids import p_ball
from reachability_utils.plot_reachable_sets import plot, project_ellipsoid_to_2d, plot_ellipse


reachable_set_data = {}

def generate_reachable_sets(file):
    ndata = 1500
    data = make_vessel_samples(ndata)
    last_step = []
    all_steps = []

    for i in range(ndata):
        last_step.append(data[i, -1])
    last_step = np.array(last_step)

    step = []
    for time_step in range(1, data.shape[1]):
        for i in range(ndata):
            for j in range(data.shape[2]):
                step.append(data[i, time_step, j])
        opt, A, b = p_ball(step, 2)
        b = b.flatten()
        xc = np.mean(step, axis=0)
        reachable_set_data[time_step] = A, b, xc

        all_steps.append(np.array(step))
        step = []

    with open(file, 'wb') as f:
        pickle.dump(reachable_set_data, f)

    # --- Plotting Code ---
    # fig, ax = plt.subplots()
    # i = 0
    # for step in reachable_set_data.values():
    #     A, b, xc = step
    #     Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(A, b, 2, 4)
    #     ax = plot_ellipse(Q_proj, c_proj, r_proj, label=f"Ellipsoid {i}", ax=ax)
    #     i+=1
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()
    # ---------------------
    
if __name__ == '__main__':
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    file = os.path.join(script_directory, 'reachable_sets_vessel.pkl')
    generate_reachable_sets(file)
