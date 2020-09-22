"""
compute_length_scales.py
==========================

Computes vertical length from Appendix B. Brethouwer 2007.
"""

import h5py
import numpy as np

from fluidsim import load_params_simul


def compute_length_scales(path_simulation, tmin=None):
    """
    Compute length scales.
    """
    # Print out
    res_out = float(path_simulation.split("NS2D.strat_")[1].split("x")[0])
    gamma_str = path_simulation.split("_gamma")[1].split("_")[0]
    if gamma_str.startswith("0"):
        gamma_out = float(gamma_str[0] + "." + gamma_str[1])
    else:
        gamma_out = float(gamma_str)

    print(f"Compute dissipation nx = {res_out} and gamma {gamma_out}..")

    # Load object parameters
    params = load_params_simul(path_simulation)

    # Load data energy spectra
    with h5py.File(path_simulation + "/spectra1D.h5", "r") as f:
        times = f["times"][...]
        kx = f["kxE"][...]
        ky = f["kyE"][...]
        spectrum1Dkx_EK_ux = f["spectrum1Dkx_EK_ux"][...]
        spectrum1Dky_EK_uy = f["spectrum1Dky_EK_uy"][...]

    # Temporal average spectra
    dt = np.median(np.diff(times))
    if not tmin:
        nb_files = 100
        tmin = np.max(times) - (nb_files * dt)
    itmin = np.argmin(abs(times - tmin))

    # Compute time average
    spectrum1Dkx_EK_ux = np.mean(spectrum1Dkx_EK_ux[itmin:, :], axis=0)
    spectrum1Dky_EK_uy = np.mean(spectrum1Dky_EK_uy[itmin:, :], axis=0)

    # Compute spectrum dealiased
    kxmax_dealiasing = params.oper.coef_dealiasing * np.max(abs(kx))
    kymax_dealiasing = params.oper.coef_dealiasing * np.max(abs(ky))

    ikx_dealiasing = np.argmin(abs(kx - kxmax_dealiasing))
    iky_dealiasing = np.argmin(abs(ky - kymax_dealiasing))

    kx = kx[0:ikx_dealiasing]
    ky = ky[0:iky_dealiasing]
    spectrum1Dkx_EK_ux = spectrum1Dkx_EK_ux[:ikx_dealiasing]
    spectrum1Dky_EK_uy = spectrum1Dky_EK_uy[:iky_dealiasing]

    # Compute delta_k
    delta_kx = np.median(np.diff(abs(kx)))
    delta_ky = np.median(np.diff(abs(ky)))

    # Compute vertical length scale
    lx = np.sum(spectrum1Dkx_EK_ux * delta_kx) / np.sum(
        kx * spectrum1Dkx_EK_ux * delta_kx
    )

    lz = np.sum(spectrum1Dky_EK_uy * delta_ky) / np.sum(
        ky * spectrum1Dky_EK_uy * delta_ky
    )

    return lx, lz


if __name__ == "__main__":
    # path_simulation = "/fsnet/project/meige/2015/15DELDUCA/DataSim/sim1920_no_shear_modes/NS2D.strat_1920x480_S2pix1.571_F07_gamma02_2018-08-14_09-59-55"

    path_simulation = "/fsnet/project/meige/2015/15DELDUCA/DataSim/sim1920_no_shear_modes/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-08-14_10-01-22"

    print(path_simulation)
    lx, lz = compute_length_scales(path_simulation)

    print("lx = ", lx)
    print("lz = ", lz)
