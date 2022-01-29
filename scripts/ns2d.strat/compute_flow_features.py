"""
compute_flow_features.py
========================
28/09/2018

"""
import os
import numpy as np
import h5py

from glob import glob
import matplotlib.pyplot as plt

from fluidsim import load_params_simul

# Argparse arguments
nx = 3840
n_files_average = 50  # Number of files to perform the time average

# Create paths
path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim"

if nx == 1920:
    directory = f"sim{nx}_no_shear_modes"
elif nx == 3840 or nx == 7680:
    directory = f"sim{nx}_modif_res_no_shear_modes"
else:
    raise ValueError(".")

path_simulations = sorted(glob(os.path.join(path_root, directory, "NS2D*")))

gammas = []
anisotropies_gammas = []
ratio_dissipations = []
for ipath, path in enumerate(path_simulations):
    params = params = load_params_simul(path)

    # Add gamma to list gammas
    gamma_str = path.split("_gamma")[1].split("_")[0]
    if gamma_str.startswith("0"):
        gammas.append(float(gamma_str[0] + "." + gamma_str[1]))
    else:
        gammas.append(float(gamma_str))

    # Compute time average ratio ux**2 / uy**2 (anisotropy)
    print("Computing anisotropy for gamma {}...".format(gammas[ipath]))
    path_phys_files = glob(path + "/state_phys_t*")
    anisotropies = []
    for path_file in path_phys_files[-n_files_average:]:
        with h5py.File(path_file, "r") as f:
            ux = f["state_phys"]["ux"][...]
            uz = f["state_phys"]["uy"][...]
            anisotropies.append(np.mean(ux**2) / np.mean(uz**2))
    anisotropies_gammas.append(np.mean(anisotropies))

    # Compute ratio D(k_x)/epsilon
    print("Computing ratio dissipation for gamma {}...".format(gammas[ipath]))
    with h5py.File(path + "/spect_energy_budg.h5", "r") as f:
        kx = f["kxE"][...]
        kz = f["kyE"][...]
        dset_dissEKu_kx = f["dissEKu_kx"]
        dset_dissEKv_kx = f["dissEKv_kx"]
        dset_dissEA_kx = f["dissEA_kx"]

        delta_kx = kx[1] - kx[0]
        delta_kz = kz[1] - kz[0]

        dissEK_kx = (
            dset_dissEKu_kx[-n_files_average:]
            + dset_dissEKv_kx[-n_files_average:]
        )
        dissEA_kx = dset_dissEA_kx[-n_files_average:]
        dissE_kx = (dissEK_kx + dissEA_kx).mean(0)
        D_kx = dissE_kx.cumsum() * delta_kx

        # Compute k_fx
        k_fx = (
            np.sin(params.forcing.tcrandom_anisotropic.angle)
            * params.forcing.nkmax_forcing
            * max(delta_kx, delta_kz)
        )
        ik_fx = np.argmin(abs(kx - k_fx))

        # Compute ratio
        ratio_dissipations.append(D_kx[ik_fx] / D_kx[-1])

fig1, ax1 = plt.subplots()
ax1.set_xlabel(r"$\gamma$")
ax1.set_ylabel(r"$U_x^2/U_z^2$")
ax1.plot(gammas, anisotropies_gammas, "ro")

fig2, ax2 = plt.subplots()
ax2.set_xlabel(r"$\gamma$")
ax2.set_ylabel(r"$D(k_{fx})/D(k_{x, max})$")
ax2.plot(gammas, ratio_dissipations, "bo")

plt.show()
