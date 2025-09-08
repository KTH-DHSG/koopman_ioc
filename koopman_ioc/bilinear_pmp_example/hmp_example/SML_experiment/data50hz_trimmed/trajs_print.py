import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

# ------------------------------ IMPORT REAL-WORLD DATA ------------------------------
filepath = 'koopman_ioc/bilinear_pmp_example/hmp_example/SML_experiment/data50hz_trimmed'
M = 48  # trajectories test0.csv ... test47.csv

x_star = []
u_star = []
trajectory_lengths = []

for i in range(M):
    filename = f'{filepath}/test{i}.csv'
    data = pd.read_csv(filename)

    x_pos = data['x_pos'].values
    y_pos = data['y_pos'].values
    orientation = data['orientation'].values
    u_linear = data['u_linear'].values
    u_angular = data['u_angular'].values

    # trim to equal length columns
    min_len = min(len(x_pos), len(y_pos), len(orientation), len(u_linear), len(u_angular))
    x_pos = x_pos[:min_len]; y_pos = y_pos[:min_len]; orientation = orientation[:min_len]
    u_linear = u_linear[:min_len]; u_angular = u_angular[:min_len]

    trajectory_lengths.append(min_len)

    states = np.vstack([x_pos, y_pos, orientation]).T
    controls = np.vstack([u_linear, u_angular]).T

    x_star.append(states)
    u_star.append(controls)

# ------------------------------ PLOTTING SETUP ------------------------------
# Indices of the 4 test trajectories you want to highlight
test_ids = [26, 28, 39, 46]
train_color = '#1f77b4'  # blue
test_color = '#d62728'   # red

fig, ax = plt.subplots(figsize=(10, 6))

# plot train trajectories (same color)
for i, traj in enumerate(x_star):
    if i in test_ids:
        continue
    ax.plot(traj[:, 0], traj[:, 1], color=train_color, linewidth=0.8)

# plot test trajectories (different color) + label inside plot
for tid in test_ids:
    traj = x_star[tid]
    ax.plot(traj[:, 0], traj[:, 1], color=test_color, linewidth=1.2)
    # label near the midpoint of each trajectory
    mid = len(traj) // 2
    ax.text(traj[mid, 0], traj[mid, 1], f'{tid}', color=test_color, fontsize=10,
            ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

ax.set_xlabel('x position')
ax.set_ylabel('y position')
ax.set_title('Trajectories')
ax.grid(True, alpha=0.3)
# No legend per requirement

# ------------------------------ INSET PANELS ------------------------------
# Paths to existing test-trajectory figures
inset_images = {
    26: 'images/R01_traj26_fullview2.png',
    28: 'images/R01_traj28_fullview2.png',
    39: 'images/R01_traj39_fullview2.png',
    46: 'images/R01_traj46_fullview2.png',
}
# Positions in axes fraction (x,y) for TL, TR, BR, BL
positions = {
    26: (0.02, 0.98),   # top-left
    28: (0.98, 0.98),   # top-right
    39: (0.98, 0.02),   # bottom-right
    46: (0.02, 0.02),   # bottom-left
}
# A reasonable zoom (adjust if needed)
zoom = 0.40

for tid, img_path in inset_images.items():
    if not os.path.isfile(img_path):
        print(f'Warning: inset image not found: {img_path}')
        continue
    arr_img = plt.imread(img_path)
    imagebox = OffsetImage(arr_img, zoom=zoom)
    ab = AnnotationBbox(imagebox, positions[tid], xycoords='axes fraction',
                        frameon=True, boxcoords='axes fraction', pad=0.02, bboxprops=dict(edgecolor='0.6'))
    ax.add_artist(ab)

plt.tight_layout()
# Save the composite
os.makedirs('images', exist_ok=True)
plt.savefig('images/trajs_composite.png', dpi=300, bbox_inches='tight')
plt.close(fig)