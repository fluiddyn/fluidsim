"""
compute_anisotropy.py
=====================
1/10/2018

Function to compute the anisotropy of a simulation.

"""
from glob import glob

import h5py
import numpy as np

from fluidsim import load_params_simul


def _compute_array_times_from_path(path_simulation):
    """
    Compute array with times from path simulation.

    Parameters
    ----------
    path_simulation : str
      Path of the simulation.
    """
    times_phys_files = []

    paths_phys_files = glob(path_simulation + "/state_phys_t*")
    for path in paths_phys_files:
        if not "=" in path.split("state_phys_t")[1].split(".nc")[0]:
            times_phys_files.append(
                float(path.split("state_phys_t")[1].split(".nc")[0])
            )
        else:
            continue
    return np.asarray(times_phys_files)


def compute_anisotropy(path_simulation, tmin=None):
    """
    It computes the anisotropy of a simulation.

    The anisotropy is defined as:

    ::math EK_ux / EK

    Parameters
    ----------
    path_simulation : str
      Path of the simulation.

    tmin : float
      Lower limit to compute the time average.
      By default it takes the last 10 files.
    """

    # Print out
    res_out = float(path_simulation.split("NS2D.strat_")[1].split("x")[0])
    gamma_str = path_simulation.split("_gamma")[1].split("_")[0]
    if gamma_str.startswith("0"):
        gamma_out = float(gamma_str[0] + "." + gamma_str[1])
    else:
        gamma_out = float(gamma_str)

    print(f"Compute anisotropy nx = {res_out} and gamma {gamma_out}..")

    # Load data energy spectra file.
    with h5py.File(path_simulation + "/spectra1D.h5", "r") as f:
        kx = f["kxE"][...]
        times = f["times"][...]
        spectrumkx_EK_ux = f["spectrum1Dkx_EK_ux"][...]
        spectrumkx_EK = f["spectrum1Dkx_EK"][...]
        spectrumkx_EK_uy = f["spectrum1Dkx_EK_uy"][...]
        dt_state_phys = np.median(np.diff(times))

    # Compute start time average itmin
    dt = np.median(np.diff(times))
    if not tmin:
        nb_files = 10
        tmin = np.max(times) - (nb_files * dt)
    itmin = np.argmin(abs(times - tmin))

    # Compute delta_kx
    params = load_params_simul(path_simulation)
    delta_kx = 2 * np.pi / params.oper.Lx

    # Compute spatial averaged energy from spectra
    EK_ux = np.sum(np.mean(spectrumkx_EK_ux[itmin:, :], axis=0) * delta_kx)
    EK = np.sum(np.mean(spectrumkx_EK[itmin:, :], axis=0) * delta_kx)

    return EK_ux / EK


if __name__ == "__main__":
    path_simulation = (
        "/fsnet/project/meige/2015/15DELDUCA/DataSim/"
        + "sim1920_no_shear_modes/NS2D.strat_1920x480_S2pix1.571_F07_gamma05_2018-08-14_10-01-22"
    )

    anisotropy = compute_anisotropy(path_simulation)
    print(f"anisotropy = {anisotropy}")
