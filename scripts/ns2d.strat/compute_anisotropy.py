"""
compute_anisotropy.py
=====================
1/10/2018

Function to compute the anisotropy of a simulation.

"""
from glob import glob

import h5py
import numpy as np

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
                float(path.split("state_phys_t")[1].split(".nc")[0]))
        else:
            continue
    return np.asarray(times_phys_files)

def compute_anisotropy(path_simulation, tmin=None):
    """
    It computes the anisotropy of a simulation.

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

    print("Compute anisotropy nx = {} and gamma {}..".format(res_out, gamma_out))

    # Compute index start average time imin.
    times = _compute_array_times_from_path(path_simulation)
    dt_state_phys = np.median(np.diff(times))

    if not tmin:
        nb_files = 10
        tmin = np.max(times) - (nb_files * dt_state_phys)
    imin = np.argmin(abs(times - tmin))

    # Compute anisotropy
    anisotropies = []
    for path in glob(path_simulation + "/state_phys_t*")[imin:]:
        with h5py.File(path, "r") as f:
            ux = f["state_phys"]["ux"].value
            uz = f["state_phys"]["uy"].value
        anisotropies.append(np.mean(ux**2 / (ux**2 + uz**2)))

    return np.mean(anisotropies)

if __name__ == "__main__":
   path_simulation = ("/fsnet/project/meige/2015/15DELDUCA/DataSim/" +
                      "sim1920_no_shear_modes/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-08-14_10-01-22")

   anisotropy = compute_anisotropy(path_simulation)
   print("anisotropy = {}".format(anisotropy))
