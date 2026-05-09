import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------- Replicator dynamics ----------
def replicator(t, u, A):
    du = np.zeros_like(u)
    avg_fit = np.dot(u, A @ u)   # u^T A u
    
    for i in range(4):
        du[i] = u[i] * ((A @ u)[i] - avg_fit)
        
    return du

# ---------- RK4 integrator ----------
def rk4(f, t_span, u0, dt, args=()):
    t0, tf = t_span
    t_values = np.arange(t0, tf + dt, dt)
    
    u = np.zeros((len(u0), len(t_values)))
    u[:, 0] = u0

    for i in range(len(t_values) - 1):
        t = t_values[i]
        ui = u[:, i]

        k1 = f(t, ui, *args)
        k2 = f(t + dt/2, ui + dt*k1/2, *args)
        k3 = f(t + dt/2, ui + dt*k2/2, *args)
        k4 = f(t + dt, ui + dt*k3, *args)

        u_next = ui + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        # ---- enforce simplex (VERY IMPORTANT) ----
        u_next = np.maximum(u_next, 0)
        u_next /= np.sum(u_next)

        u[:, i+1] = u_next

    return t_values, u



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

# ---------- initial condition ----------
#change initial condition to get closer to the unstable limit cycle
# this is already very close
initx1 = 0.27608
initx2 = 0.22958
initx3 = 0.24799
u0 = np.array([initx1, initx2, initx3, 1 - initx1 - initx2 - initx3])



# ---------- RK4 solve ----------
dt = 0.01   # try smaller dt for higher accuracy

t, u = rk4(
    replicator,
    (0, 2000),
    u0,
    dt,
    args=(A,)
)

u1, u2, u3, u4 = u

# ---------- 3D plot ----------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

split = 2243  #time period of the limit cycle chage according to the plot to get time period of one cycle
ax.plot(u1[:split], u2[:split], u3[:split], color='red', lw=1, marker='o', markersize=3)



ax.scatter(u1[0], u2[0], u3[0], color='blue', s=80)  #initial point
ax.scatter(u1[split], u2[split], u3[split], color='black', s=80)  #final point

ax.set_xlabel("u1")
ax.set_ylabel("u2")
ax.set_zlabel("u3")
ax.set_title(f"3D trajectory (mu = {mu})")

# compute x4 from simplex constraint
x4 = 1 - u1[:split] - u2[:split] - u3[:split]
data = np.column_stack((u1[:split], u2[:split], u3[:split], x4))

np.savetxt(
    "trajectory.txt",
    data,
    fmt="%.8f",
    header="x1 x2 x3 x4",
    comments=""
)

plt.show()