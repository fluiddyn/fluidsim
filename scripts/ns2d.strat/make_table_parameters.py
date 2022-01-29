"""
make_table_parameters.py
=========================
28/09/2018

"""
import os
import h5py
import numpy as np

from glob import glob

from fluidsim import load_sim_for_plot

# Argparse arguments
nx = 3840
MAKE_TABLE = True

# Parameters script
n_files_tmean = 100

# Create path
path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim"

if nx == 1920:
    directory = "sim1920_no_shear_modes"
elif nx == 3840:
    directory = "sim3840_modif_res_no_shear_modes"

path_simulations = sorted(glob(os.path.join(path_root, directory, "NS2D*")))

if MAKE_TABLE:
    path_table = (
        "/home/users/calpelin7m/"
        + f"Phd/docs/Manuscript/buoyancy_reynolds_table_n{nx}.tex"
    )

    to_print = (
        "\\begin{table}[h]\n"
        "\\centering \n"
        "\\begin{tabular}{cccc} \n"
        "\\toprule[1.5pt] \n"
        + "\\bm{$\\gamma$} & \\bm{$F_h$} & \\bm{$Re_8$} & \\bm{$\\mathcal{R}$} \\\\ \n"
        "\\midrule\\ \n"
    )

for path in path_simulations:

    # Load object simulations
    sim = load_sim_for_plot(path)

    # Compute gamma from path
    gamma_str = path.split("_gamma")[1].split("_")[0]
    if gamma_str.startswith("0"):
        gamma_table = float(gamma_str[0] + "." + gamma_str[1])
    else:
        gamma_table = float(gamma_str)

    # Compute mean kinetic dissipation
    dict_spatial = sim.output.spatial_means.load()

    times = dict_spatial["t"]
    epsK_tot = dict_spatial["epsK_tot"]
    epsK_tmean = np.mean(epsK_tot[-n_files_tmean:], axis=0)
    print("epsilon", epsK_tmean)
    # Compute horizontal scale as Appendix B. Brethouwer (2007)
    path_spectra = path + "/spectra1D.h5"

    with h5py.File(path_spectra, "r") as f:
        times_spectra = f["times"][...]
        kx = f["kxE"][...]
        spectrum1Dkx_EK_ux = f["spectrum1Dkx_EK_ux"][...]

    spectrum1Dkx_EK_ux = np.mean(spectrum1Dkx_EK_ux[-100:, :], axis=0)
    ## Remove modes with dealiasing
    kxmax_dealiasing = sim.params.oper.coef_dealiasing * np.max(abs(kx))
    ikxmax = np.argmin(abs(kx - kxmax_dealiasing))
    kx = kx[:ikxmax]
    spectrum1Dkx_EK_ux = spectrum1Dkx_EK_ux[:ikxmax]
    delta_kx = sim.oper.deltakx
    lx = np.sum(spectrum1Dkx_EK_ux * delta_kx) / np.sum(
        kx * spectrum1Dkx_EK_ux * delta_kx
    )
    print("lx", lx)
    # Compute eta_8
    eta_8 = (sim.params.nu_8**3 / epsK_tmean) ** (1 / 22)
    print("eta_8", eta_8)

    # Compute Re_8
    Re_8 = (lx / eta_8) ** (22 / 3)
    print("Re_8", Re_8)

    # Compute horizontal Froude
    F_h = ((epsK_tmean / lx**2) ** (1 / 3)) * (1 / sim.params.N)
    print("F_h", F_h)

    # Reynolds buoyancy 8
    Rb8 = Re_8 * F_h**8
    print("Rb8", Rb8)

    if MAKE_TABLE:
        to_print += "{} & {:.4f} & {:.4e} & {:.4e} \\\\ \n".format(
            gamma_table, F_h, Re_8, Rb8
        )

if MAKE_TABLE:
    with open(path_table, "w") as f:
        to_print += "\\end{tabular} \n" "\\end{table}"
        f.write(to_print)
