import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# === User-defined parameter ===
N = 2060  # Change this to select how many last rows you want to plot for one cycle

# === Load data ===
data = np.loadtxt("replicator_trajectory.txt", skiprows=1)  # skip header
x1, x2, x3 = data[-N:, 0], data[-N:, 1], data[-N:, 2]  # last N rows


# === Plot ==========
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1, x2, x3, lw=2, marker='o', markersize=2,color='red')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
plt.tight_layout()


np.savetxt("limit_cycle_data.txt", data[-N:,],fmt="%.4f", header="x1 x2 x3", comments='')
# print(data[-N:,])



# Payoff matrix
A = np.array([
    [-0.022, -0.829, -0.405, 5.5],
    [0.45, -0.211, 0.825, -1.531],
    [1.295, -0.008, -0.226, -0.96],
    [0.10, 0.25, 0.05, 0]
])



# Replicator dynamics
def replicator(x):
    payoff = A @ x
    avg_payoff = x @ payoff
    dxdt = x * (payoff - avg_payoff)
    return dxdt

# RK4 step
def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


#=================================================
#trajectory number 1
#=================================================
# Initial condition
x1int, x2int, x3int = 0.190, 0.1430, 0.1578
x0 = np.array([x1int, x2int, x3int, 1 - x1int - x2int - x3int])
# print([x1[0], x2[0], x3[0], 1-x1[0]-x2[0]-x3[0]])
# print(x0)


# Parameters
time_run = 2000
dt = 0.01
num_steps = int(time_run / dt)

# Allocate memory
trajectory = np.zeros((num_steps + 1, 4))
trajectory[0] = x0

# Integrate
x = x0.copy()
for i in range(1, num_steps + 1):
    x = rk4_step(replicator, x, dt)
    trajectory[i] = x
    sum_traj=trajectory[i,0]+trajectory[i,1]+trajectory[i,2]+trajectory[i,3]



ax.plot(trajectory[:,0],trajectory[:,1], trajectory[:,2], lw=0.5, marker='o', markersize=1,color='green')
ax.plot(trajectory[-3000:,0],trajectory[-3000:,1], trajectory[-3000:,2], lw=0.5, marker='o', markersize=1,color='red')
# #==================================
# #trajectory number 2
# #==================================
# Initial condition
x1int, x2int, x3int = 0.3, 0.2, 0.35
x0 = np.array([x1int, x2int, x3int, 1 - x1int - x2int - x3int])
# print([x1[0], x2[0], x3[0], 1-x1[0]-x2[0]-x3[0]])
# print(x0)
# Parameters
time_run = 2000
dt = 0.01
num_steps = int(time_run / dt)

# Allocate memory
trajectory = np.zeros((num_steps + 1, 4))
trajectory[0] = x0

# Integrate
x = x0.copy()
for i in range(1, num_steps + 1):
    x = rk4_step(replicator, x, dt)
    trajectory[i] = x
    sum_traj=trajectory[i,0]+trajectory[i,1]+trajectory[i,2]+trajectory[i,3]


ax.plot(trajectory[:,0],trajectory[:,1], trajectory[:,2], lw=0.5, marker='o', markersize=1,color='blue')
ax.plot(trajectory[-3000:,0],trajectory[-3000:,1], trajectory[-3000:,2], lw=0.5, marker='o', markersize=1,color='red')
ax.set_xlabel(r"$x_1$", fontsize=40, labelpad=16)
ax.set_ylabel(r"$x_2$", fontsize=40, labelpad=16)
ax.set_zlabel(r"$x_3$", fontsize=40, labelpad=16)
ax.tick_params(labelsize=25)
#===============================================
ax.view_init(elev=20, azim=60)
plt.savefig("trajectory_view_e30_a45.png", dpi=300)
plt.show()
