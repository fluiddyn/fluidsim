"""
compute_reynolds_froude.py
==========================
1/10/2018

"""

import h5py
import numpy as np

from fluidsim import load_params_simul


def _compute_epsilon_from_path(path_simulation, tmin=None):
    """
    Computes the mean dissipation from tmin
    """
    # Load data dissipation
    with open(path_simulation + "/spatial_means.txt", "r") as f:
        lines = f.readlines()

    lines_t = []
    lines_epsK = []

    for il, line in enumerate(lines):
        if line.startswith("time ="):
            lines_t.append(line)
        if line.startswith("epsK ="):
            lines_epsK.append(line)

    nt = len(lines_t)
    t = np.empty(nt)
    epsK = np.empty(nt)

    for il in range(nt):
        line = lines_t[il]
        words = line.split()
        t[il] = float(words[2])

        line = lines_epsK[il]
        words = line.split()
        epsK[il] = float(words[2])

    # Compute start time average itmin
    dt_spatial = np.median(np.diff(t))

    if not tmin:
        nb_files = 100
        tmin = np.max(t) - (dt_spatial * nb_files)

    itmin = np.argmin((abs(t - tmin)))

    return np.mean(epsK[itmin:], axis=0)


def _compute_lx_from_path(path_simulation, tmin=None):
    """
    Compute horizontal length from path using appendix B. Brethouwer 2007
    """

    # Load parameters from simulation
    params = load_params_simul(path_simulation)

    with h5py.File(path_simulation + "/spectra1D.h5", "r") as f:
        times_spectra = f["times"][...]
        kx = f["kyE"][...]
        spectrum1Dkx_EK_ux = f["spectrum1Dkx_EK_ux"][...]

    # Compute time average spectra
    dt = np.median(np.diff(times_spectra))

    if not tmin:
        nb_files = 100
        tmin = np.max(times_spectra) - (dt * nb_files)
    itmin = np.argmin(abs(times_spectra - tmin))

    spectrum1Dkx_EK_ux = np.mean(spectrum1Dkx_EK_ux[-itmin:, :], axis=0)

    # Remove modes dealiased
    ikxmax = np.argmin(abs(kx - (np.max(kx) * params.oper.coef_dealiasing)))
    kx = kx[:ikxmax]
    spectrum1Dkx_EK_ux = spectrum1Dkx_EK_ux[:ikxmax]
    delta_kx = np.median(np.diff(kx))

    # Compute horizontal length scale Brethouwer 2007
    return np.sum(spectrum1Dkx_EK_ux * delta_kx) / np.sum(
        kx * spectrum1Dkx_EK_ux * delta_kx
    )


def compute_buoyancy_reynolds(path_simulation, tmin=None):
    """
    Compute the buoyancy Reynolds number.
    """
    params = load_params_simul(path_simulation)
    epsK = _compute_epsilon_from_path(path_simulation, tmin=tmin)
    lx = _compute_lx_from_path(path_simulation, tmin=tmin)

    F_h = ((epsK / lx**2) ** (1 / 3)) * (1 / params.N)

    eta_8 = (params.nu_8**3 / epsK) ** (1 / 22)
    Re_8 = (lx / eta_8) ** (22 / 3)

    R_b = Re_8 * F_h**8
    return F_h, Re_8, R_b


# path_simulation = "/fsnet/project/meige/2015/15DELDUCA/DataSim/sim1920_no_shear_modes/NS2D.strat_1920x480_S2pix1.571_F07_gamma02_2018-08-14_09-59-55"
# F_h, Re_8, R_b = compute_buoyancy_reynolds(path_simulation)
# print("F_h", F_h)
# print("Re_8", Re_8)
# print("R_b", R_b)
