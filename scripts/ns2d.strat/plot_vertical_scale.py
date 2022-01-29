"""
plot_vertical_scale.py
=======================

Plots vertical scale.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fluidsim import load_params_simul
from compute_length_scales import compute_length_scales
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
# directories = directories[1:]
for directory in directories:
    paths_simulations += sorted(glob(os.path.join(path_root, directory, "NS2D*")))

# Define lists
froudes = []
reynoldsb = []
reynolds8 = []
anisotropies = []
dissipations = []
markers = []
lzs_billant = []

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

    # Compute Billant length scale
    ux_rms = []
    path_phys_files = glob(path + "/state_phys_t*")
    for path_file in path_phys_files[-10:-1]:
        with h5py.File(path_file, "r") as f:
            ux = f["state_phys"]["ux"][...]
        ux_rms.append(np.sqrt(np.mean(ux**2)))

    # Load parameters
    params = load_params_simul(path)

    lzs_billant.append(l_z * params.N / np.mean(ux_rms))

# Parameters figures
set_rcparams(fontsize=14, for_article=True)
fig, ax = plt.subplots()
ax.set_xlabel(r"$\mathcal{R}_8$", fontsize=18)
ax.set_ylabel(r"$l_z N / U$", fontsize=18)
ax.set_xscale("log")
ax.set_yscale("linear")
ax.set_xlim([1e-10, 1e8])
ax.set_ylim([0, 1.2e1])

# Plot...
for _f, _r, _a, _d, _m, _l in zip(
    froudes, reynoldsb, anisotropies, dissipations, markers, lzs_billant
):
    scatter = ax.scatter(
        _r, _l, s=100, c=_d, vmin=0, vmax=1, marker=_m, alpha=0.7
    )

# Plot horizontal line
ax.axhline(y=1, color="k", linestyle="--")

# Colorbar...
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cax.tick_params(labelsize=14)
fig.colorbar(scatter, cax=cax)
ax.text(
    1e7,
    0.013,
    r"$\log_{10} \left(\frac{k_{x, 1/2}}{k_{x, f}}\right)$",
    fontsize=12,
)

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
    marker="^",
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

SAVE = False
if SAVE:
    path_save = (
        "/fsnet/project/meige/2015/15DELDUCA/notebooks/figures/"
        + "vertical_length_scale.png"
    )
    fig.savefig(path_save, format="png", bbox_inches="tight")

plt.show()
