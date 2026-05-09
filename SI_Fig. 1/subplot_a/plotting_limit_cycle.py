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
N = 2243  # Change this to select how many last rows you want to plot for one cycle

# === Load data ===
data = np.loadtxt("trajectory.txt", skiprows=1)  # skip header
x1, x2, x3 = data[-N:, 0], data[-N:, 1], data[-N:, 2]  # last N rows

# === Plot ===
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1, x2, x3, lw=2, marker='o', markersize=6,color='red')

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')

plt.tight_layout()


np.savetxt("limit_cycle_data.txt", data[-N:,],fmt="%.4f", header="x1 x2 x3", comments='')
# print(data[-N:,])



# ---------- parameter ----------
# mu = 71/48 - 1e-4   # VERY close to bifurcation
mu =1.479

# ---------- matrix A ----------
A = np.array([
    [-2,   -5,   -0.5, 7.5],
    [-0.5, -1,   -mu, 1.5+mu],
    [-1,   -0.5, -1, 2.5],
    [0,   0, 0, 0]
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
x1int, x2int, x3int = 0.226, 0.273, 0.25
x0 = np.array([x1int, x2int, x3int, 1 - x1int - x2int - x3int])
# print([x1[0], x2[0], x3[0], 1-x1[0]-x2[0]-x3[0]])
# print(x0)

# Parameters
time_run = 100000
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



ax.plot(trajectory[:,0],trajectory[:,1], trajectory[:,2], lw=1.0, marker='o', markersize=2,color='green')

# #==================================
# #trajectory number 2
# #==================================
# Initial condition
x1int, x2int, x3int = 0.27608, 0.22958, 0.24799
x0 = np.array([x1int, x2int, x3int, 1 - x1int - x2int - x3int])
# print([x1[0], x2[0], x3[0], 1-x1[0]-x2[0]-x3[0]])
# print(x0)
# Parameters
time_run = 100000
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



ax.plot(trajectory[:,0],trajectory[:,1], trajectory[:,2], lw=1.0, marker='o', markersize=2,color='blue')


ax.set_xlabel(r"$x_1$", fontsize=35, labelpad=15)
ax.set_ylabel(r"$x_2$", fontsize=35, labelpad=15)
ax.set_zlabel(r"$x_3$", fontsize=35, labelpad=15)
ax.set_xlim(0.2, 0.3)
ax.set_ylim(0.2, 0.3)
ax.set_zlim(0.2, 0.3)

ax.set_xticks([0.2,0.25, 0.30])
ax.set_yticks([0.2,0.25, 0.30])
ax.set_zticks([0.2,0.25, 0.30])

minor_ticks = np.linspace(0.2, 0.30, 7)  # more grid lines

ax.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.5)

ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(minor_ticks, minor=True)
ax.set_zticks(minor_ticks, minor=True)

ax.tick_params(axis='x', pad=9)
ax.tick_params(axis='y', pad=9)
ax.tick_params(axis='z', pad=9)
ax.tick_params(labelsize=25)
#===============================================
ax.view_init(elev=20, azim=60)
plt.savefig("trajectory_view_e30_a45.png", dpi=300)
plt.show()
