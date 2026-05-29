import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pprint as pp

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
        # Vessel:
        # epsilon for vessel reachable tube:  0.1069038165032687
        # epsilon for vessel ellipsoid t=1:  0.045339298531341304
        # epsilon for vessel ellipsoid t=2:  0.044308993920390914
        # epsilon for vessel ellipsoid t=3:  0.045339298531341304
        # epsilon for vessel ellipsoid t=4:  0.050388731511159586
        # epsilon for vessel ellipsoid t=5:  0.034621301651395646
        epsilons = {
            1: 0.045339298531341304,
            2: 0.044308993920390914,
            3: 0.045339298531341304,
            4: 0.050388731511159586,
            5: 0.034621301651395646
        }
        epsilon_ct = 0
        for ct in entry['robustness'][-2:]:
            epsilon_ct += epsilons[ct]

        rob_integers.extend([epsilon_ct])

    # --- 3. Compute and Print Evaluations ---
    print("=== Evaluation Results ===")

    # Runtimes
    avg_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)
    print(f"Runtime -> Average: {avg_runtime:.5f}s, Std Dev: {std_runtime:.5f}s")

    # Robustness Integers Count
    print("\n Epsilon ct (mean, min, max):")
    print(np.mean(rob_integers))
    print(np.min(rob_integers))
    print(np.max(rob_integers))

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


with open('examples/vessel_navigation/vessel_multi_runs.pkl', 'rb') as f:
    data = pickle.load(f)

# timesteps for run 0
print("\n--- Characteristic Timesteps 1---")
print("\n--- Run 0 Timestep 1 ---")
pp.pprint(data[0]['log'][1])
print("\n--- Run 0 Timestep 2 ---")
pp.pprint(data[0]['log'][2])
print("\n--- Run 0 Timestep 3 ---")
pp.pprint(data[0]['log'][3])

print("\n--- Characteristic Timesteps 2---")
print("\n--- Run 0 Timestep 25 ---")
pp.pprint(data[0]['log'][25])
print("\n--- Run 0 Timestep 26 ---")
pp.pprint(data[0]['log'][26])
print("\n--- Run 0 Timestep 27 ---")
pp.pprint(data[0]['log'][27])

print("\n--- Characteristic Timesteps 5---")
print("\n--- Run 0 Timestep 35 ---")
pp.pprint(data[0]['log'][35])
print("\n--- Run 0 Timestep 85 ---")
pp.pprint(data[0]['log'][85])
print("\n--- Run 0 Timestep 105 ---")
pp.pprint(data[0]['log'][105])