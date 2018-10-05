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

def _get_resolution_from_dir(path_simulation):
    return path_simulation.split("NS2D.strat_")[1].split("x")[0]

# Create path simulations
path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim"
directories = ["sim960_no_shear_modes",
               "sim1920_no_shear_modes",
               "sim1920_modif_res_no_shear_modes",
               "sim3840_modif_res_no_shear_modes"]

# directories = ["sim960_no_shear_modes",
#                "sim1920_no_shear_modes"]

paths_simulations = []
for directory in directories:
    paths_simulations += sorted(glob(os.path.join(path_root, directory, "NS2D*")))

froudes = []
reynoldsb = []
anisotropies = []
dissipations = []
markers = []

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
    res = _get_resolution_from_dir(path)

    froudes.append(F_h)
    reynoldsb.append(R_b)
    anisotropies.append(anisotropy)
    dissipations.append(dissipation)

    if res == "960":
        markers.append("o")
    elif res == "1920":
        markers.append("s")
    elif res == "3840":
        markers.append("^")

    print("F_h", F_h)
    print("Re_8", Re_8)
    print("R_b", R_b)

for _f, _r, _a, _d, _m in zip(froudes, reynoldsb, anisotropies, dissipations, markers):
    scatter = ax.scatter(_f, _r, s=250 * (_a**2), c=_d, vmin=0, vmax=0.3, marker=_m)
# plt.show()

# areas = 250 * np.asarray(anisotropies)**2
# scatter = ax.scatter(froudes, reynoldsb, s=areas, c=dissipations, alpha=0.7, vmin=0, vmax=0.3)
ax.scatter(max(froudes), 1e2 * min(reynoldsb), s=250 * np.asarray(0.5)**2, c="red")
ax.scatter(max(froudes), min(reynoldsb), s=250 * np.asarray(1.0)**2, c="red")
ax.text(0.12, 90 * min(reynoldsb), "anisotropy=0", fontsize=12, color="r")
ax.text(0.12, min(reynoldsb), "anisotropy=1", fontsize=12, color="r")
fig.colorbar(scatter)

# Legend
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

blue_star = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=8, label=r'$n_x = 960$')
red_square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                          markersize=8, label=r'$n_x = 1920$')
purple_triangle = mlines.Line2D([], [], color='red', marker='^', linestyle='None',
                          markersize=8, label=r'$n_x = 3840$')

ax.legend(handles=[blue_star, red_square, purple_triangle],
          loc="upper center",
          bbox_to_anchor=(0.5,1.1),
          borderaxespad=0.,
          ncol=len(markers),
          handletextpad=0.1,
          fontsize=12)
plt.show()
