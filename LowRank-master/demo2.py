import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from functools import partial

# ==============================
# 1. 工具函数（和你原来的风格一致）
# ==============================

def GetBulk(N):
    """随机高斯 bulk（这里我们等会儿设成 0，只保留低秩部分）"""
    return np.random.normal(0.0, np.sqrt(1.0 / N), size=(N, N))

def Integrate(X, t, J, I):
    """
    连续时间 RNN 动力学:
        dX/dt = -X + J * tanh(X) + I
    X: (N,)
    J: (N, N)
    I: (N,)
    """
    return -X + J.dot(np.tanh(X)) + I

def SimulateActivity(t, x0, J, I):
    """
    使用 scipy.integrate.odeint 积分 RNN 动力学
    t: 时间数组
    x0: 初始状态 (N,)
    J: 连接矩阵 (N,N)
    I: 外部输入 (N,)
    """
    print(" ** Simulating... **")
    traj = scipy.integrate.odeint(partial(Integrate, J=J, I=I), x0, t)
    return traj  # 形状 (len(t), N)

# ==============================
# 2. 构造一个 rank-1 低秩网络，产生近似 line attractor
# ==============================

np.random.seed(0)

N = 1000              # 单位数（用 500 就够看结构了，比 7000 快很多）
g = 0.3              # 这里先关掉 bulk，只保留 rank-1 低秩结构

# 构造一个向量 m（同时作为 n），使得 m^T m / N ≈ 1，这样沿 m 的方向近似中性
m = np.random.normal(0.0, 1.0, size=N)
m = m / np.sqrt(np.mean(m**2))   # 归一化到平均平方 ~ 1
n = m.copy()

# 低秩部分 M = m n^T / N
M = np.outer(m, n) / N

# 可选的随机 bulk，这里设为 0（如果你想要更真实一点，可以设小 g）
R = g * GetBulk(N)

# 总连接矩阵
J = R + M

# ==============================
# 3. 沿着 attractor 采样一维状态 (通过不同强度的输入 I_scale * m)
# ==============================

# 时间设置：给足够长时间收敛
T = 50.0
dt = 0.1
t = np.arange(0.0, T, dt)

# 初始状态
x0 = np.zeros(N)

# 在不同的输入强度 I_scale 下，网络会收敛到沿着 m 方向的一条“线”上的不同点
n_points = 31
I_scales = np.linspace(-1.0, 1.0, n_points)

# 用来存储 steady-state 的 firing rate 和 1D 坐标 v
R_states = np.zeros((n_points, N))  # 每一行是一个固定点的 tanh(x*)
v_coords = np.zeros(n_points)       # 每个固定点的 1D 坐标 v

for idx, alpha in enumerate(I_scales):
    print(f"\n=== Simulate fixed point for I_scale = {alpha:.2f} ===")
    I = alpha * m          # 外部输入沿着 m 方向
    traj = SimulateActivity(t, x0, J, I)
    x_star = traj[-1]      # 最后时刻近似看作 fixed point
    r_star = np.tanh(x_star)
    R_states[idx, :] = r_star

    # 用 m 的投影作为 attractor 上的 1D 坐标 v
    v = np.dot(m, r_star) / N
    v_coords[idx] = v

# 按照 v 的大小排序（方便后续画曲线）
order = np.argsort(v_coords)
v_sorted = v_coords[order]
R_sorted = R_states[order, :]

# ==============================
# 4. 定义 3 个任务 φ(v) = v, v^2, v^4
#    并用最小二乘反推出对应的 readout 向量 w1, w2, w3
# ==============================

phi1 = v_sorted
phi2 = v_sorted**2
phi3 = v_sorted**4

# 设计矩阵 A: (M, N)，对应 R_sorted / N
A = R_sorted / N

# 最小二乘求解 A w ≈ φ(v)
w1, *_ = np.linalg.lstsq(A, phi1, rcond=None)
w2, *_ = np.linalg.lstsq(A, phi2, rcond=None)
w3, *_ = np.linalg.lstsq(A, phi3, rcond=None)

# 用求出的 w1, w2, w3 在同样的状态上做预测，检查拟合效果
z1 = A.dot(w1)  # 应该接近 phi1
z2 = A.dot(w2)  # 应该接近 phi2
z3 = A.dot(w3)  # 应该接近 phi3

# ==============================
# 5. 可视化 1：沿着 attractor 的函数拟合效果
# ==============================

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

axes[0].plot(v_sorted, phi1, 'k-', label='target v')
axes[0].plot(v_sorted, z1, 'r--', label='readout')
axes[0].set_title(r'$\varphi_1(v) = v$')
axes[0].set_xlabel('v'); axes[0].set_ylabel('output')
axes[0].legend()

axes[1].plot(v_sorted, phi2, 'k-', label='target v^2')
axes[1].plot(v_sorted, z2, 'r--', label='readout')
axes[1].set_title(r'$\varphi_2(v) = v^2$')
axes[1].set_xlabel('v'); axes[1].set_ylabel('output')

axes[2].plot(v_sorted, phi3, 'k-', label='target v^4')
axes[2].plot(v_sorted, z3, 'r--', label='readout')
axes[2].set_title(r'$\varphi_3(v) = v^4$')
axes[2].set_xlabel('v'); axes[2].set_ylabel('output')

plt.tight_layout()
plt.show()

# ==============================
# 6. 可视化 2：在表示空间中画出 attractor，
#    用颜色表示不同任务的几何结构（类似 NeurIPS 图 5）
# ==============================

# 对 R_sorted 做 PCA（只取前两主成分）
R_centered = R_sorted - R_sorted.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)
PC = R_centered.dot(Vt[:2].T)  # 形状 (M, 2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

titles = [r'$\varphi_1(v)=v$', r'$\varphi_2(v)=v^2$', r'$\varphi_3(v)=v^4$']
values = [z1, z2, z3]

for ax, z, title in zip(axes, values, titles):
    sc = ax.scatter(PC[:, 0], PC[:, 1], c=z, cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    plt.colorbar(sc, ax=ax)

plt.tight_layout()
plt.show()