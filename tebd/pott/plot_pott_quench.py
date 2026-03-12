import numpy as np
import matplotlib.pyplot as plt

# List of data files (update with your actual paths)
files = [
    "dqpt_potts_L50_f010.0_f10.0_julia.dat",
    "dqpt_potts_L50_f010.0_f10.0_python.dat",
]

# Option 1: Manually define labels (clear and robust)
labels = [
    r"$julia$",
    r"$python$",
  
]



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
plt.savefig("dqpt_potts.png", dpi=150, bbox_inches="tight")
plt.show()