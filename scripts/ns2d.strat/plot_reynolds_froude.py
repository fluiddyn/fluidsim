"""
plot_reynolds_froude.py
========================
1/10/2018

Makes plot buoyancy reynolds Vs Froude.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from compute_anisotropy import compute_anisotropy
from compute_ratio_dissipation import compute_ratio_dissipation
from compute_reynolds_froude import compute_buoyancy_reynolds
from fluiddyn.output.rcparams import set_rcparams


# Create path simulations
path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim"
directories = ["sim1920_no_shear_modes", "sim1920_modif_res_no_shear_modes"]
paths_simulations = []
for directory in directories:
    paths_simulations += sorted(glob(os.path.join(path_root, directory, "NS2D*")))

froudes = []
reynoldsb = []
anisotropies = []
dissipations = []

set_rcparams(fontsize=14, for_article=True)

fig, ax = plt.subplots()
ax.set_xlabel(r"$F_h$")
ax.set_ylabel(r"$\mathcal{R}$")
ax.set_xscale("log")
ax.set_yscale("log")
fig.text(0.8, 4e-7, r"$\frac{D(k_{fx})}{D(k_x)}$", fontsize=16)

for path in paths_simulations:
    F_h, Re_8, R_b = compute_buoyancy_reynolds(path)
    anisotropy = compute_anisotropy(path)
    dissipation = compute_ratio_dissipation(path)

    froudes.append(F_h)
    reynoldsb.append(R_b)
    anisotropies.append(anisotropy)
    dissipations.append(dissipation)

    print("F_h", F_h)
    print("Re_8", Re_8)
    print("R_b", R_b)

areas = 500 * np.asarray(anisotropies)**2
scatter = ax.scatter(froudes, reynoldsb, s=areas, c=dissipations, alpha=0.7)
ax.scatter(0.7, 1e-4, s=500 * np.asarray(0.5)**2, c="red")
ax.text(0.64, 1e-5, "isotropy", fontsize=14, color="r")
fig.colorbar(scatter)
plt.show()
