import numpy as np
import matplotlib.pyplot as plt

# Use LaTeX-style math text for labels
plt.rcParams["text.usetex"] = False  # set True if you have LaTeX installed
plt.rcParams["font.size"] = 14

# x-axis grid
x = np.linspace(-3, 5, 600)

# Define example functions with similar shapes
def f(x):
    # tall narrow peak on the right
    return 1.0 * np.exp(-0.7 * (x - 2.5) ** 2)

def mu(x):
    # broader bump a bit to the left - increased amplitude to make it more visible
    return 0.9 * np.exp(-0.25 * (x - 1.5) ** 2)

def nu(x):
    # constant line (like p(x) in the original) - increased value to make it more visible
    return 0.6 * np.ones_like(x)

f_vals = f(x)
mu_vals = mu(x)
nu_vals = nu(x)

# Avoid division by zero in the ratio
eps = 1e-6
ratio_vals = nu_vals / (mu_vals + eps)

# Create figure with two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

# -------- Left panel: f(x), μ(x), ν(x) --------
ax1.plot(x, f_vals, color="purple", linewidth=2.5, label=r"$f(x)$")
ax1.plot(x, mu_vals, color="red",   linewidth=2.5, label=r"$\mu(x)$")
ax1.plot(x, nu_vals, color="blue",  linewidth=2.5, label=r"$\nu(x)$")

# Annotations
ax1.text(2.6, f(2.5)+0.05,  r"$f(x)$", color="purple", ha="left", va="bottom")
ax1.text(1.4, mu(1.4)+0.05, r"$\mu(x)$", color="red",   ha="center", va="bottom")
ax1.text(-2.5, nu(-2.5)+0.02, r"$\nu(x)$", color="blue", ha="left", va="bottom")

ax1.set_xlabel("x")
ax1.set_ylabel("density")
ax1.set_xlim(x.min(), x.max())
ax1.set_ylim(0, max(f_vals.max(), mu_vals.max(), nu_vals.max()) * 1.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(loc="upper right", frameon=False)

# -------- Right panel: f(x), μ(x), ν(x)/μ(x) --------
# Faded ν(x) and μ(x)
ax2.plot(x, mu_vals, color="red",  linewidth=2.5, alpha=0.25)
ax2.plot(x, nu_vals, color="blue", linewidth=2.5, alpha=0.25)

# Main f(x) and ratio ν(x)/μ(x)
ax2.plot(x, f_vals,      color="purple", linewidth=2.5, label=r"$f(x)$")
ax2.plot(x, ratio_vals,  color="magenta", linewidth=2.5, label=r"$\nu(x)/\mu(x)$")

# Annotations
ax2.text(2.6, f(2.5)+0.05,     r"$f(x)$", color="purple", ha="left", va="bottom")
ax2.text(3.5, ratio_vals[-80], r"$\nu(x)/\mu(x)$", color="magenta",
         ha="left", va="center")

ax2.set_xlabel("x")
ax2.set_xlim(x.min(), x.max())
ax2.set_ylim(0, max(f_vals.max(), ratio_vals[np.isfinite(ratio_vals)].max()) * 1.1)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()