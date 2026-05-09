import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
# Define binary red-blue colormap
from matplotlib.colors import ListedColormap
from scipy.integrate import simpson
from mpl_toolkits.mplot3d import Axes3D
# Map: -1 to blue, 1 to red
cmap = ListedColormap(['#425fff', '#fe5f10'])

#=====================================================
# Load limit cycle data
data_xhat = np.loadtxt("limit_cycle_data.txt", skiprows=1)
xhat0=data_xhat[0]
print(data_xhat[0])
tol=0.000001 ##tolerence

# Payoff matrix
A = np.array([
    [-0.022, -0.829, -0.505, 5.5],
    [0.45, -0.211, 1.325, -1.531],
    [1.295, -0.008, -0.326, -0.96],
    [0.10, 0.25, 0.1, 0]
])


# Replicator dynamics 
def replicator(x):
    payoff = A @ x
    avg_payoff = x @ payoff
    return x * (payoff - avg_payoff)

# RK4 step
def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# Function to compute OESS
def compute_OESS(e1, e2, e3):
    #seting up the initial neighborhood
    x1int = xhat0[0] + e1
    x2int = xhat0[1] + e2
    x3int = xhat0[2] + e3
    x0 = np.array([x1int, x2int, x3int, 1 - x1int - x2int - x3int])
    x = x0.copy()
    dt = 0.01
    trajectory = np.zeros((2*len(data_xhat), 4))
    trajectory[0] = x0

    #generating the neirghborhod sequence
    for t in range(1,2*len(data_xhat)):
        x = rk4_step(replicator, x, dt)
        trajectory[t] = x

    #finding the average of the neihborbood x: intrgral_t0-T^t0 xdt for each t0
    x_avg_t0minusTtoT=np.zeros((len(data_xhat), 4))
    for t in range(len(data_xhat)):
        x_avg_t0minusTtoT[t] = simpson(trajectory[t:len(data_xhat)+t], dx=dt, axis=0)
    

    #final condition whose integral is to be done from 0 to T
    #xhat(t0)A<x>(t0)-x(t0)A<x>(t0)
    condition_data=np.zeros(len(data_xhat))
    for t0 in range(len(data_xhat)):
        condition_data[t0]=(data_xhat[t0]-trajectory[len(data_xhat)+t0]) @ A @ x_avg_t0minusTtoT[t0]
    
    #final integral from 0 to T 
    oess_val = simpson(condition_data, dx=dt)


    # Return sign with 2-level distinction
    if oess_val > tol:
        return 1
    else:
        return -1


# Grid for eps1 eps2 and eps3
eps_range = np.linspace(-0.00001, 0.00001, 20)   # increase resolution as needed
E1, E2, E3 = np.meshgrid(eps_range, eps_range, eps_range, indexing='ij')
shape = E1.shape

OESS_grid = np.zeros(shape)
for i in range(shape[0]):
    print(i)
    for j in range(shape[1]):
        for k in range(shape[2]):
            e1 = E1[i,j,k]
            e2 = E2[i,j,k]
            e3 = E3[i,j,k]
            OESS_grid[i,j,k] = compute_OESS(e1, e2, e3)



# Save to file
np.save("OESS_grid.npy", OESS_grid)
np.save("eps_range.npy", eps_range)
print("OESS grid saved!")