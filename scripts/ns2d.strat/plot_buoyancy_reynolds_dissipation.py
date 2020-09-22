"""
plot_buoyancy_reynolds_dissipation.py
=====================================

Plots the buoyancy reynolds Vs dissipation.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from mpl_toolkits.axes_grid1 import make_axes_locatable

from glob import glob
from flow_features import get_features_from_sim, _get_resolution_from_dir
from fluiddyn.output.rcparams import set_rcparams

path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim"
directories = [
    "sim960_no_shear_modes",
    "sim960_no_shear_modes_transitory",
    "sim1920_no_shear_modes",
    "sim1920_modif_res_no_shear_modes",
    "sim3840_modif_res_no_shear_modes",
    "sim7680_modif_res_no_shear_modes",
]


paths_simulations = []
# directories = [directories[0]]
for directory in directories:
    paths_simulations += sorted(glob(os.path.join(path_root, directory, "NS2D*")))

# Define lists
froudes = []
reynoldsb = []
reynolds8 = []
anisotropies = []
dissipations = []
markers = []

for path in paths_simulations:

    anisotropy, ratio_diss, F_h, Re_8, R_b, l_x, l_z = get_features_from_sim(path)
    res = _get_resolution_from_dir(path)

    # Append data to lists
    if res == "960":
        markers.append("o")
    elif res == "1920":
        markers.append("s")
    elif res == "3840":
        markers.append("^")
    elif res == "7680":
        markers.append("d")

    froudes.append(F_h)
    reynoldsb.append(R_b)
    reynolds8.append(Re_8)
    anisotropies.append(np.log10(2 * anisotropy) / np.log10(2))
    dissipations.append(np.log10(ratio_diss))

# Parameters figures
set_rcparams(fontsize=14, for_article=True)
fig, ax = plt.subplots()
ax.set_xlabel(r"$\mathcal{R}_8$", fontsize=18)
ax.set_ylabel(r"$D$", fontsize=18)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1e-9, 1e9])

# Plot...
for _r, _d, _a, _m in zip(reynoldsb, dissipations, anisotropies, markers):
    scatter = ax.scatter(_r, _d, s=100, c=_a, vmin=0, vmax=1, marker=_m)

# Colorbar...
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cax.tick_params(labelsize=14)
fig.colorbar(scatter, cax=cax)
# ax.text(1e7, 0.013, r"$\log_{10} \left(\frac{k_{x, 1/2}}{k_{x, f}}\right)$", fontsize=12)

# Legend...
blue_star = mlines.Line2D(
    [],
    [],
    color="red",
    marker="o",
    linestyle="None",
    markersize=8,
    label=r"$n_x = 960$",
)
red_square = mlines.Line2D(
    [],
    [],
    color="red",
    marker="s",
    linestyle="None",
    markersize=8,
    label=r"$n_x = 1920$",
)
purple_triangle = mlines.Line2D(
    [],
    [],
    color="red",
    marker="^",
    linestyle="None",
    markersize=8,
    label=r"$n_x = 3840$",
)
diamond = mlines.Line2D(
    [],
    [],
    color="red",
    marker="d",
    linestyle="None",
    markersize=8,
    label=r"$n_x = 7680$",
)

ax.legend(
    handles=[blue_star, red_square, purple_triangle, diamond],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    borderaxespad=0.0,
    ncol=len(markers),
    handletextpad=0.1,
    fontsize=14,
)

# Add text...
# ax.text(1e-6, 1e-1,
#         r"$A = \frac{\log_{10} \left( 2E_{k,x} / E_k \right)}{\log_{10}(2)}$",
#         fontsize=20)

SAVE = True
if SAVE:
    path_save = (
        "/fsnet/project/meige/2015/15DELDUCA/notebooks/figures/"
        + "buoyancy_reynolds_dissipation.png"
    )
    fig.savefig(path_save, format="png", bbox_inches="tight")


plt.show()
