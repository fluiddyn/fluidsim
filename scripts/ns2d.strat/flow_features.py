"""
flow_features.py
================

It computes and saves the anisotropy, dissipation, horizontal Froude, Reynolds 8
and buoyancy Reynolds.

"""

import os

from glob import glob

from compute_anisotropy import compute_anisotropy
from compute_reynolds_froude import compute_buoyancy_reynolds
from compute_ratio_dissipation import compute_ratio_dissipation
from compute_length_scales import compute_length_scales


def compute_flow_features_from_sim(path_simulation, SAVE=False):
    """
    Computes anisotropy, dissipation, horizontal Froude, Reynolds 8
    and buoyancy Reynolds.

    Parameters
    ----------
    path_simulation : str
      Path of the simulation to compute the flow features.

    SAVE : bool
      Saves the results in the path_directory of the simulation.

    """
    if SAVE and os.path.exists(
        os.path.join(path_simulation, "flow_features.txt")
    ):
        pass
    else:
        # Computes all the flow features
        anisotropy = compute_anisotropy(path_simulation)
        ratio_diss = compute_ratio_dissipation(path_simulation)
        F_h, Re_8, R_b = compute_buoyancy_reynolds(path_simulation)
        l_x, l_z = compute_length_scales(path_simulation)

        if SAVE:
            path_save = os.path.join(path_simulation, "flow_features.txt")
            to_print = (
                f"anisotropy = {anisotropy} \n"
                + f"ratio_diss = {ratio_diss} \n"
                + f"F_h = {F_h} \n"
                + f"Re_8 = {Re_8} \n"
                + f"R_b = {R_b} \n"
                + f"l_x = {l_x} \n"
                + f"l_z = {l_z} \n"
            )

            # Checks if the file flow_features.txt exists
            if os.path.exists(path_save):
                pass
            else:
                with open(path_save, "w") as f:
                    f.write(to_print)
        return anisotropy, ratio_diss, F_h, Re_8, R_b, l_x, l_z


def get_features_from_sim(path_simulation):
    """
    Gets features of a simulation
    """
    path_file = os.path.join(path_simulation, "flow_features.txt")

    if not os.path.exists(path_file):
        raise ValueError(f"{path_file} does not exist.")

    with open(path_file, "r") as f:
        lines = f.readlines()

    for il, line in enumerate(lines):
        if line.startswith("anisotropy = "):
            anisotropy = float(line.split()[2])
        if line.startswith("ratio_diss = "):
            ratio_diss = float(line.split()[2])
        if line.startswith("F_h = "):
            F_h = float(line.split()[2])
        if line.startswith("Re_8 = "):
            Re_8 = float(line.split()[2])
        if line.startswith("R_b = "):
            R_b = float(line.split()[2])
        if line.startswith("l_x = "):
            l_x = float(line.split()[2])
        if line.startswith("l_z = "):
            l_z = float(line.split()[2])

    return anisotropy, ratio_diss, F_h, Re_8, R_b, l_x, l_z


def _get_resolution_from_dir(path_simulation):
    return path_simulation.split("NS2D.strat_")[1].split("x")[0]


if __name__ == "__main__":
    # path_simulation = ("/fsnet/project/meige/2015/15DELDUCA/DataSim/" +
    #                 "sim1920_no_shear_modes/NS2D.strat_1920x480_S2pix1.571_F07_gamma05_2018-08-14_10-01-22")

    # anisotropy, ratio_diss, F_h, Re_8, R_b = \
    #                 compute_flow_features_from_sim(path_simulation, SAVE=False)

    # Compute flow features from all simulations...

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
    #     directories = [directories[0]]
    for directory in directories:
        paths_simulations += sorted(
            glob(os.path.join(path_root, directory, "NS2D*"))
        )

    # To compute flow features in all files
    for path_simulation in paths_simulations:
        compute_flow_features_from_sim(path_simulation, SAVE=True)

    # # To remove all files flow_features
    # for path in paths_simulations:
    #     os.remove(os.path.join(path, "flow_features.txt"))
