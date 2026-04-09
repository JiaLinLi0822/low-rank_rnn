# Rank-2 low-rank RNN: simulate a line attractor and plot (fast, low-rank matvec)
# J*y is computed via rank-2 factors to avoid heavy N×N matmul.

import numpy as np
import matplotlib.pyplot as plt
import os

def phi(x):
    return np.tanh(x)

def rand_orthonormal(N, K, rng):
    M = rng.normal(size=(N, K))
    Q, _ = np.linalg.qr(M, mode='reduced')
    return Q[:, :K]

def build_lowrank_modes(N, alpha=0.999, beta=0.8, seed=0):
    rng = np.random.default_rng(seed)
    U = rand_orthonormal(N, 2, rng)
    u1, u2 = U[:,0], U[:,1]
    m1 =  np.sqrt(alpha*N) * u1
    n1 =  np.sqrt(alpha*N) * u1
    m2 =  np.sqrt(beta*N)  * u2
    n2 =  np.sqrt(beta*N)  * u2
    return m1, m2, n1, n2

def J_times_y_lowrank(y, m1, m2, n1, n2, N):
    s1 = float(n1 @ y) / N
    s2 = float(n2 @ y) / N
    return m1 * s1 + m2 * s2

def simulate_lowrank(N, m1, m2, n1, n2, T=400.0, dt=1.0, tau=10.0, x0=None, record_step=2, seed=None):
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0 = rng.normal(size=N)
    nsteps = int(np.floor(T/dt))
    rec_idx = np.arange(0, nsteps, record_step)
    Kappa = np.empty((2, len(rec_idx)))
    x = x0.copy()
    ri = 0
    for t in range(nsteps):
        y = phi(x)
        Jy = J_times_y_lowrank(y, m1, m2, n1, n2, N)
        x += (dt/tau) * (-x + Jy)
        if (t % record_step) == 0:
            y = phi(x)
            Kappa[0, ri] = float(n1 @ y) / N
            Kappa[1, ri] = float(n2 @ y) / N
            ri += 1
    return Kappa, rec_idx

# ----- Demo parameters (very fast) -----
N     = 200
alpha = 0.999    # near-marginal along mode-1 => slow/line-like
beta  = 0.80     # contracting along mode-2
m1, m2, n1, n2 = build_lowrank_modes(N, alpha=alpha, beta=beta, seed=123)

n_trials = 3
kappa_trajs = []
time_axes   = []

for i in range(n_trials):
    x0 = 2.0 * np.random.default_rng(500+i).normal(size=N)
    K, rec_idx = simulate_lowrank(N, m1, m2, n1, n2, T=500.0, dt=1.0, tau=10.0,
                                  x0=x0, record_step=2, seed=99+i)
    kappa_trajs.append(K)
    time_axes.append(rec_idx * 1.0)

# 1) κ1 over time
plt.figure(figsize=(7, 4.2))
for i in range(n_trials):
    plt.plot(time_axes[i], kappa_trajs[i][0, :])
plt.xlabel("time")
plt.ylabel("kappa1 (slow / marginal)")
plt.title("κ1 vs. time (line-attractor axis)")
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/kappa1_time.png", dpi=150)
plt.show()

# 2) κ2 over time
plt.figure(figsize=(7, 4.2))
for i in range(n_trials):
    plt.plot(time_axes[i], kappa_trajs[i][1, :])
plt.xlabel("time")
plt.ylabel("kappa2 (contracting)")
plt.title("κ2 vs. time (contracting axis)")
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/kappa2_time.png", dpi=150)
plt.show()

# 3) Phase plot: κ2 vs κ1
plt.figure(figsize=(6, 6))
for i in range(n_trials):
    plt.plot(kappa_trajs[i][0, :], kappa_trajs[i][1, :])
plt.xlabel("kappa1")
plt.ylabel("kappa2")
plt.title("Phase plot: κ2 vs κ1 (collapse to 1D manifold)")
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/phase_k2_vs_k1.png", dpi=150)
plt.show()

# Print numerical start/end for quick check
for i in range(n_trials):
    K = kappa_trajs[i]
    start = K[:, 0]
    end = K[:, -1]
    print(f"Trial {i+1}: start (k1={start[0]:.3f}, k2={start[1]:.3f}) -> end (k1={end[0]:.3f}, k2={end[1]:.3f})")