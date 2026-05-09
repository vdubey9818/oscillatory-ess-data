import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
start_time = time.time()
# ---------- Replicator system ----------
def replicator(t, u, A):
    du = np.zeros_like(u)
    Au = A @ u
    avg_fit = np.dot(u, Au)   # u^T A u
    
    for i in range(4):
        du[i] = u[i] * (Au[i] - avg_fit)
        
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



def compute_poincare(u1, u2, u3, c):
    points = []
    for i in range(len(u3) - 1):
        if u3[i] < c and u3[i+1] >= c:
            alpha = (c - u3[i]) / (u3[i+1] - u3[i])
            x1 = u1[i] + alpha * (u1[i+1] - u1[i])
            x2 = u2[i] + alpha * (u2[i+1] - u2[i])
            points.append([x1, x2])
    return np.array(points)

def random_simplex_point():
    x = np.random.rand(4)
    return x / np.sum(x)

# ---------- Poincaré section ----------
c = 0.24799
dt = 0.01

init_conditions = [
    ( 0.226, 0.273, 'green'),
    (0.228, 0.269, 'blue')
]

all_points = []

for x1_init, x2_init, color in init_conditions:

    x3_init = 0.251   # slightly off the section (important!)
    x4_init = 1 - x1_init - x2_init - x3_init

    u0 = np.array([x1_init, x2_init, x3_init, x4_init])

    t, u = rk4(
        replicator,
        (0, 200000),
        u0,
        dt,
        args=(A,)
    )

    u1, u2, u3, u4 = u

    pts = compute_poincare(u1, u2, u3, c)

    if len(pts) > 0:
        all_points.append((pts, color))

plt.figure(figsize=(8,8))

for pts, color in all_points:
    if len(pts) > 1:

        x = pts[:,0]
        y = pts[:,1]
        scale_vis = 5
        plt.scatter(x, y, s=50, alpha=1.0, color=color)

        # draw arrows x_i → x_{i+1}
        for i in range(len(x) - 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]

plt.scatter(0.22779015, 0.27079800 , s=80, alpha=1.0, color='red')
plt.xlabel(f"$x_1$", fontsize=50)
plt.ylabel(f"$x_2$",fontsize=50)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(0.18,0.25)
plt.ylim(0.24,0.30)
plt.grid()
plt.tight_layout()
plt.savefig(f'poincare_map.png',dpi=300)
#======================================================
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime:.6f} seconds")
#======================================================
plt.show()
