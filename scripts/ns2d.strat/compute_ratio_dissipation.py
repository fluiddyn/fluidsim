"""
compute_ratio_dissipation.py
============================
02/10/2018
"""

import h5py
import numpy as np

from fluidsim import load_params_simul

def compute_ratio_dissipation(path_simulation, tmin=None):
    """
    Compute ratio dissipation from path simulation.
    """
    # Print out
    res_out = float(path_simulation.split("NS2D.strat_")[1].split("x")[0])
    gamma_str = path_simulation.split("_gamma")[1].split("_")[0]
    if gamma_str.startswith("0"):
        gamma_out = float(gamma_str[0] + "." + gamma_str[1])
    else:
        gamma_out = float(gamma_str)

    print("Compute dissipation nx = {} and gamma {}..".format(res_out, gamma_out))

    # Load object parameters
    params = load_params_simul(path_simulation)

    with h5py.File(path_simulation + "/spect_energy_budg.h5", "r") as f:
        times = f["times"].value
        kx = f["kxE"].value
        kz = f['kyE'].value
        dset_dissEKu_kx = f['dissEKu_kx']
        dset_dissEKv_kx = f['dissEKv_kx']
        dset_dissEA_kx = f["dissEA_kx"]

        # Compute itmin time average
        if not tmin:
            nb_files = 10
            dt = np.median(np.diff(times))
            tmin = np.max(times) - (nb_files * dt)
        itmin = np.argmin(abs(times - tmin))

        # Compute dissipation curve
        delta_kx = np.median(np.diff(kx))
        delta_kz = np.median(np.diff(abs(kz)))
        dissEK_kx = dset_dissEKu_kx[-itmin:] + dset_dissEKv_kx[-itmin:]
        dissEA_kx = dset_dissEA_kx[-itmin:]
        dissE_kx = (dissEK_kx + dissEA_kx).mean(0)
        D_kx = dissE_kx.cumsum() * delta_kx

    # Compute k_fx
    k_fx = (np.sin(params.forcing.tcrandom_anisotropic.angle) *
            params.forcing.nkmax_forcing * max(delta_kx, delta_kz))
    ik_fx = np.argmin(abs(kx - k_fx ))

    return D_kx[ik_fx] / D_kx[-1]
