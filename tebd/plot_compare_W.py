import numpy as np
import matplotlib.pyplot as plt

# List of data files (update with your actual paths)
files = [
    "imbalance_fk_z3_L12_g0.3_W0.5_U2.dat",
    "imbalance_fk_z3_L12_g0.3_W1.0_U2.dat",
    "imbalance_fk_z3_L12_g0.3_W2.0_U2_julia.dat",
    "imbalance_fk_z3_L12_g0.3_W2.0_U2.dat",
    "imbalance_fk_z3_L12_g0.3_W4.0_U2.dat",
    "imbalance_fk_z3_L12_g0.3_W8.0_U2.dat",
    "imbalance_fk_z3_L12_g0.3_W8.0_U2_julia.dat",
]

# Option 1: Manually define labels (clear and robust)
labels = [
    r"$W=0.5,\ U_2$",
    r"$W=1.0,\ U_2$",
    r"$W=2.0,\ U_2$",
    r"$W=2.0,\ U_2$ (Julia)",
    r"$W=4.0,\ U_2$",
    r"$W=8.0,\ U_2$",
    r"$W=8.0,\ U_2$ (Julia)",
]

# Option 2: Automatically generate labels from filenames (uncomment if preferred)
# def make_label(fname):
#     # Remove common prefix and suffix, replace underscores
#     lbl = fname.replace("imbalance_fk_z3_L12_g0.3_", "").replace(".dat", "")
#     lbl = lbl.replace("_", ", ")
#     return f"${lbl}$"   # wrap in math mode for LaTeX style
# labels = [make_label(f) for f in files]

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 3))

# Loop over files and plot
for fname, lbl in zip(files, labels):
    data = np.loadtxt(fname)           # reads two columns by default
    t = data[:, 0]
    I = data[:, 1]
    ax.plot(t, I, label=lbl)

# Labels (using LaTeX syntax if desired)
ax.set_xlabel(r"$t\,[J^{-1}]$")
ax.set_ylabel(r"$I(t)$")
ax.legend(loc="upper right")           # or loc="best"

# Save and show
plt.savefig("compare_imbalance_W1_W8.png", dpi=150, bbox_inches="tight")
plt.show()