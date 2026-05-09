import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
start_time = time.time()

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

# Initial condition
x1int, x2int, x3int = 0.2190, 0.4430, 0.1578
x0 = np.array([x1int, x2int, x3int, 1 - x1int - x2int - x3int])

# Parameters
time_run = 100000
dt = 0.01
num_steps = int(time_run / dt)

# Allocate memory
trajectory = np.zeros((num_steps + 1, 4))
trajectory[0] = x0

# Integrate
start_time = time.time()
x = x0.copy()
for i in range(1, num_steps + 1):
    x = rk4_step(replicator, x, dt)
    trajectory[i] = x

# guess the period roughly by seein the plot
N_period = 2169 
N_last = 10 * N_period
data = trajectory

np.savetxt("replicator_trajectory.txt", data[-N_last:,], fmt="%.4f", header="x1 x2 x3", comments='')

# Unpack x1, x2, x3 from the last N_last points
x1, x2, x3 = data[:, 0], data[:, 1], data[:, 2]


# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1, x2, x3, lw=1.5)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title("Limit Cycle Trajectory (Last 10 Periods)")
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime:.6f} seconds")
plt.tight_layout()
plt.show()
