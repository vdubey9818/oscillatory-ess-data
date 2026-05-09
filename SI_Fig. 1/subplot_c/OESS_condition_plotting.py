# ------------------------------
# plot_OESS_grid.py
# ------------------------------
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load saved grid and epsilon range
OESS_grid = np.load("OESS_grid.npy")
eps_range = np.load("eps_range.npy")

# Prepare mesh
E1, E2, E3 = np.meshgrid(eps_range, eps_range, eps_range, indexing='ij')

# Plotting
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

alpha = 0.3
colors = np.zeros(OESS_grid.shape + (4,), dtype=float)

# Red for OESS=1
mask_pos = (OESS_grid == 1)
colors[mask_pos] = [1, 0, 0, alpha]

# Blue for OESS=-1
mask_neg = (OESS_grid == -1)
colors[mask_neg] = [0, 0, 1, alpha]

# Voxels: include both
ax.voxels(mask_pos | mask_neg, facecolors=colors)

# Label axes
ax.set_xlabel(r'$\epsilon_1$')
ax.set_ylabel(r'$\epsilon_2$')
ax.set_zlabel(r'$\epsilon_3$')

N = len(eps_range)
ticks = [0, (N-1)//2, N-1]
labels = [" ", "0", " "]

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_zticklabels(labels)
ax.set_xlabel(r"$\epsilon_1$", fontsize=40, labelpad=10)
ax.set_ylabel(r"$\epsilon_2$", fontsize=40, labelpad=10)
ax.set_zlabel(r"$\epsilon_3$", fontsize=40, labelpad=10)
ax.tick_params(labelsize=30)
ax.view_init(elev=200, azim=300)
# plt.tight_layout()
plt.savefig("OESS_view.png", dpi=300)
plt.show()
