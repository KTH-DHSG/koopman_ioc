import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress scientific notation when printing NumPy arrays
np.set_printoptions(precision=5, suppress=True)

## ------------------------------ IMPORT REAL-WORLD DATA ------------------------------

filepath = 'koopman_ioc/bilinear_pmp_example/hmp_example/SML_experiment/data50hz_trimmed'

# Number of trajectories to use (training set)
M = 47+1

# Initialize lists to hold data
x_star = []
u_star = []
start_pt = []
trajectory_lengths = []

# Read the training data from test0.csv to the M trajectory
for i in range(M):
    filename = f'{filepath}/test{i}.csv'
    data = pd.read_csv(filename)
    x_pos = data['x_pos'].values
    y_pos = data['y_pos'].values
    orientation = data['orientation'].values
    u_linear = data['u_linear'].values
    u_angular = data['u_angular'].values

    # Ensure all arrays are the same length
    min_len = min(len(x_pos), len(u_linear), len(u_angular))
    x_pos = x_pos[:min_len]
    y_pos = y_pos[:min_len]
    orientation = orientation[:min_len]
    u_linear = u_linear[:min_len]
    u_angular = u_angular[:min_len]
    
    trajectory_lengths.append(min_len)

    states = np.vstack([x_pos, y_pos, orientation]).T  # shape (min_len, 3)
    controls = np.vstack([u_linear, u_angular]).T      # shape (min_len, 2)

    start_pt.append(states[0, :])  # initial state

    x_star.append(states)
    u_star.append(controls)

start_pt = np.array(start_pt)  # shape (M, 3)

# Print the number of datapoints for each trajectory
for i, length in enumerate(trajectory_lengths):
    print(f"Trajectory {i}: {length} datapoints")

# Plot the trajectories
plt.figure(figsize=(10, 6))
for i, traj in enumerate(x_star):
    x_pos = traj[:, 0]
    y_pos = traj[:, 1]
    plt.plot(x_pos, y_pos, label=f'Trajectory {i} ({trajectory_lengths[i]} pts)')
plt.legend()
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Trajectories')
plt.grid()
plt.show()