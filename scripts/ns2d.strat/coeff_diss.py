"""
coeff_diss.py
=============

Computes the optimal dissipation coefficient.

Solver ns2d.strat

# To do by hand:
1. Go to directory of the first simulation
2. `from fluidsim.util.util import modif_resolution_from_dir`
3. `modif_resolution_from_dir(coef_modif_resol=3/2)`
4. Change path in `coeff_diss.py`
"""

import os
import h5py
import shutil
import numpy as np
import matplotlib.pyplot as plt
import argparse


from math import pi
from glob import glob
from copy import deepcopy as _deepcopy

from fluidsim import load_sim_for_plot
from fluidsim.solvers.ns2d.strat.solver import Simul
from fluiddyn.util import mpi

from ns2dstrat_lmode import make_parameters_simulation, modify_parameters

### PARAMETERS ###
parser = argparse.ArgumentParser()
parser.add_argument("gamma", type=float)
args = parser.parse_args()


# CONDITIONS
nb_wavenumbers_y = 16
threshold_ratio = 1e1
min_factor = 0.7

# SIMULATION
gamma = args.gamma
F = np.sin(pi / 4)  # F = omega_l / N
sigma = (
    1  # sigma = omega_l / (pi * f_cf); f_cf freq time correlation forcing in s-1
)
nu_8 = 1e-16
NO_SHEAR_MODES = False
t_end = 8.0

PLOT_FIGURES = False
###################


def load_mean_spect_energy_budg(sim, tmin=0, tmax=1000):
    """
    Loads data spect_energy_budget.
    It computes the mean between tmin and tmax.
    """

    with h5py.File(sim.output.spect_energy_budg.path_file, "r") as f:
        times = f["times"][...]
        kxE = f["kxE"][...]
        kyE = f["kyE"][...]
        dset_dissEK_kx = f["dissEK_kx"][...]
        dset_dissEA_kx = f["dissEA_kx"][...]
        dset_dissEK_ky = f["dissEK_ky"][...]
        dset_dissEA_ky = f["dissEA_ky"][...]

        imin_plot = np.argmin(abs(times - tmin))
        imax_plot = np.argmin(abs(times - tmax))

        dset_dissE_kx = dset_dissEK_kx + dset_dissEA_kx
        dissE_kx = dset_dissE_kx[imin_plot : imax_plot + 1].mean(0)

        dset_dissE_ky = dset_dissEK_ky + dset_dissEA_ky
        dissE_ky = dset_dissE_ky[imin_plot : imax_plot + 1].mean(0)

    return kxE, kyE, dissE_kx, dissE_ky


def get_state_from_sim(sim):
    """Returns the state from a simulation."""
    # Take the state
    b_fft = sim.state.get_var("b_fft")
    rot_fft = sim.state.get_var("rot_fft")

    return rot_fft, b_fft


def compute_diff(idx_dealiasing, idy_dealiasing, dissE_kx, dissE_ky):
    idx_diss_max = np.argmax(abs(dissE_kx))
    idy_diss_max = np.argmax(abs(dissE_ky))

    # Computes difference
    diff_x = idx_dealiasing - idx_diss_max
    diff_y = idy_dealiasing - idy_diss_max

    diff = np.argmax([diff_x, diff_y])

    if diff == 0:
        print(
            "diff_x = idx_dealiasing -  idx_diss_max",
            idx_dealiasing - idx_diss_max,
        )
        diff = idx_dealiasing - idx_diss_max

    else:
        print(
            "diff_y = idy_dealiasing -  idy_diss_max",
            idy_dealiasing - idy_diss_max,
        )
        diff = idy_dealiasing - idy_diss_max

    return diff, idx_diss_max, idy_diss_max


def normalization_initialized_field(sim, coef_norm=1e-4):
    """Normalizes the initialized field. (ONLY if nx != ny)"""
    if sim.params.oper.nx != sim.params.oper.ny:

        if not sim.params.forcing.key_forced == "ap_fft":
            raise ValueError("sim.params.forcing.key_forced should be ap_fft.")

        KX = sim.oper.KX
        cond = KX == 0.0

        ux_fft = sim.state.get_var("ux_fft")
        uy_fft = sim.state.get_var("uy_fft")
        b_fft = sim.state.get_var("b_fft")

        ux_fft[cond] = 0.0
        uy_fft[cond] = 0.0
        b_fft[cond] = 0.0

        # Compute energy after ux_fft[kx=0] uy_fft[kx=0] b_fft[kx=0]
        ek_fft = (np.abs(ux_fft) ** 2 + np.abs(uy_fft) ** 2) / 2
        ea_fft = ((np.abs(b_fft) / params.N) ** 2) / 2
        e_fft = ek_fft + ea_fft
        energy_before_norm = sim.output.sum_wavenumbers(e_fft)

        # Compute scale energy forcing
        Lx = sim.params.oper.Lx
        Lz = sim.params.oper.Ly

        nkmax_forcing = params.forcing.nkmax_forcing
        nkmin_forcing = params.forcing.nkmin_forcing

        k_f = ((nkmax_forcing + nkmin_forcing) / 2) * max(
            2 * pi / Lx, 2 * pi / Lz
        )
        energy_f = params.forcing.forcing_rate ** (2 / 7) * (2 * pi / k_f) ** 7

        coef = np.sqrt(coef_norm * energy_f / energy_before_norm)

        ux_fft *= coef
        uy_fft *= coef
        b_fft *= coef

        rot_fft = sim.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        sim.state.init_statespect_from(rot_fft=rot_fft, b_fft=b_fft)

        sim.state.statephys_from_statespect()
    else:
        pass
    return sim


def _compute_ikdiss(dissE_kx, dissE_ky):
    idx_diss_max = np.argmax(abs(dissE_kx))
    idy_diss_max = np.argmax(abs(dissE_ky))

    return idx_diss_max, idy_diss_max


def _compute_ikmax(kxE, kyE):
    idx_dealiasing = np.argmin(abs(kxE - sim.oper.kxmax_dealiasing))
    idy_dealiasing = np.argmin(abs(kyE - sim.oper.kymax_dealiasing))

    return idx_dealiasing, idy_dealiasing


def compute_delta_ik(kxE, kyE, dissE_kx, dissE_ky):
    idx_diss_max, idy_diss_max = _compute_ikdiss(dissE_kx, dissE_ky)
    idx_dealiasing, idy_dealiasing = _compute_ikmax(kxE, kyE)

    idx_target = idx_dealiasing - (nb_wavenumbers_x - 1)
    idy_target = idy_dealiasing - (nb_wavenumbers_y - 1)

    delta_ikx = idx_target - idx_diss_max
    delta_iky = idy_target - idy_diss_max

    return idx_target, idy_target, delta_ikx, delta_iky


def compute_energy_spatial(sim):
    """Compute energy without energy in shear modes"""
    dict_spatial = sim.output.spatial_means.load()
    E = dict_spatial["E"] - dict_spatial["E_shear"]
    t = dict_spatial["t"]

    PK_tot = dict_spatial["PK_tot"]
    PA_tot = dict_spatial["PA_tot"]
    P_tot = PK_tot + PA_tot
    return E, t, P_tot


def modify_factor(sim):
    params = _deepcopy(sim.params)

    nu_8_old = params.nu_8
    params_old = sim.params
    sim_old = sim

    params.nu_8 = params.nu_8 * factor
    params.init_fields.type = "in_script"
    params.time_stepping.t_end = t_end

    return params, nu_8_old


def write_to_file(path, to_print, mode="a"):
    with open(path, mode) as f:
        f.write(to_print)


def make_float_value_for_path(value):
    value_not = str(value)
    if "." in value_not:
        return value_not.split(".")[0] + "_" + value_not.split(".")[1]
    else:
        return value_not


def check_dissipation():
    ratio_x = dissE_kx[idx_diss_max] / dissE_kx[idx_dealiasing - 1]
    ratio_y = dissE_ky[idy_diss_max] / dissE_ky[idy_dealiasing - 1]

    idx_target, idy_target, delta_ikx, delta_iky = compute_delta_ik(
        kxE, kyE, dissE_kx, dissE_ky
    )

    if time_total > 1000:
        print("The stationarity has not " + f"reached after {it} simulations.")
        should_I_stop = "non_stationarity"

    if ratio_x > threshold_ratio and ratio_y > threshold_ratio:

        if delta_ikx > 0 and delta_iky > 0:

            if delta_ikx > delta_iky:
                print("limited by y")
                norm = idy_dealiasing
                delta_ikmin = delta_iky + nb_wavenumbers_y // 4
            else:
                print("limited by x")
                norm = idx_dealiasing
                delta_ikmin = delta_ikx + nb_wavenumbers_x // 4

            factor = max((1 - (delta_ikmin / norm)) ** 1.5, 0.5)
            should_I_stop = False

        else:
            print(f"Checking stationarity... with nu8 = {nu_8_old}")
            E, t, P_tot = compute_energy_spatial(sim)
            ratio = np.mean(np.diff(E[2:]) / np.diff(t[2:]))
            if (ratio / injection_energy_0) < 0.5 and abs(
                nu_8_old - params.nu_8
            ) / params.nu_8 < 0.05:
                print(f"Stationarity is reached.\n nu_8 = {params.nu_8}")
                factor = 1.0
                should_I_stop = True
            else:
                should_I_stop = False
                factor = 1.0
    else:

        factor = 1 + (1 / min(ratio_x, ratio_y))
        should_I_stop = False

    return factor, should_I_stop


#################################################

gamma_not = make_float_value_for_path(gamma)

# Create directory in path
path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim/Coef_Diss"
name_directory = f"Coef_Diss_gamma{gamma_not}"
path = os.path.join(path_root, name_directory)

if mpi.rank == 0 and not os.path.exists(path):
    os.mkdir(path)

# Check list simulations in directory
paths_sim = sorted(glob(os.path.join(path, "NS*")))

# Write in .txt file
path_file_write = os.path.join(path, "results.txt")

PLOT_FIGURES = PLOT_FIGURES and mpi.rank == 0

# Write temporal results in .txt file
path_file_write2 = os.path.join(path, "results_check_nu8.txt")
if mpi.rank == 0:
    if os.path.exists(path_file_write2):
        os.remove(path_file_write2)

if len(paths_sim) == 0:
    # Make FIRST SIMULATION
    params = make_parameters_simulation(
        gamma, F, sigma, nu_8, t_end=t_end, NO_SHEAR_MODES=NO_SHEAR_MODES
    )
    sim = Simul(params)
    sim = normalization_initialized_field(sim)
    sim.time_stepping.start()
else:
    # Look for path largest resolution
    new_file = glob(paths_sim[-1] + "/State_phys*")[-1]
    path_file = glob(new_file + "/state_phys*")[0]

    # Compute resolution from path
    res_str = os.path.basename(new_file).split("_")[-1]

    sim = load_sim_for_plot(paths_sim[-1])

    params = _deepcopy(sim.params)

    params.oper.nx = int(res_str.split("x")[0])
    params.oper.ny = int(res_str.split("x")[1])

    params.init_fields.type = "from_file"
    params.init_fields.from_file.path = path_file

    params.time_stepping.t_end += t_end

    params.NEW_DIR_RESULTS = True

    modify_parameters(params)

    sim = Simul(params)
    sim.time_stepping.start()


# Parameters condition
nb_wavenumbers_x = nb_wavenumbers_y * (sim.params.oper.nx // sim.params.oper.ny)

# Creation time and energy array
time_total = 0
time_total += sim.time_stepping.t

energy, t, P_tot = compute_energy_spatial(sim)
energy = np.mean(energy[len(energy) // 2 :])
injection_energy_0 = P_tot[2]

energies = []
viscosities = []
energies.append(energy)
viscosities.append(sim.params.nu_8)

# Write results into temporal file
if mpi.rank == 0:
    to_print = (
        "####\n"
        "t = {:.4e} \n"
        "E = {:.4e} \n"
        "nu8 = {:.4e} \n"
        "factor = {:.4e} \n"
    ).format(time_total, energy, sim.params.nu_8, 1)

    write_to_file(path_file_write2, to_print, mode="w")

# Compute the data spectra energy budget
kxE, kyE, dissE_kx, dissE_ky = load_mean_spect_energy_budg(
    sim, tmin=2.0, tmax=1000
)
idx_diss_max, idy_diss_max = _compute_ikdiss(dissE_kx, dissE_ky)
idx_dealiasing, idy_dealiasing = _compute_ikmax(kxE, kyE)
idx_target, idy_target, delta_ikx, delta_iky = compute_delta_ik(
    kxE, kyE, dissE_kx, dissE_ky
)

# Plot dissipation
if PLOT_FIGURES:
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("$D(k_y)$")
    ax.set_xlabel("$k_y$")
    ax.set_ylabel("$D(k_y)$")
    ax.plot(
        kyE,
        dissE_ky,
        label="nu8 = {:.2e}, diff = {}".format(
            sim.params.nu_8, abs(idy_diss_max - idy_dealiasing)
        ),
    )
    ax.plot(sim.oper.kymax_dealiasing, 0, "xr")
    ax.axvline(x=sim.oper.deltaky * idy_target, color="k")

    fig2, ax2 = plt.subplots()
    ax2.set_title("$D(k_x)$")
    ax2.set_xlabel("$k_x$")
    ax2.set_ylabel("$D(k_x)$")
    ax2.plot(
        kxE,
        dissE_kx,
        label="nu8 = {:.2e}, diff = {}".format(
            sim.params.nu_8, abs(idx_diss_max - idx_dealiasing)
        ),
    )
    ax2.plot(sim.oper.kxmax_dealiasing, 0, "xr")
    ax2.axvline(x=sim.oper.deltakx * idx_target, color="k")

    ax.legend()
    ax2.legend()

    fig.canvas.draw()
    fig2.canvas.draw()

    plt.pause(1e-3)

    # Dissipation vs time
    fig3, ax3 = plt.subplots()
    ax3.set_xlabel("times")
    ax3.set_ylabel(r"$\nu_8$")

    ax3.plot(time_total, viscosities[-1], ".")

    # Energy Vs time
    fig4, ax4 = plt.subplots()
    ax4.plot(time_total, energy, ".")
    ax4.set_xlabel("times")
    ax4.set_ylabel("Energy")

    # Factor Vs time
    fig5, ax5 = plt.subplots()
    ax5.plot(time_total, 1, ".")
    ax5.set_xlabel("times")
    ax5.set_ylabel("Factor")


it = 0
p = 1
# Check ...
while True:

    if mpi.rank == 0:
        factor, should_I_stop = check_dissipation()
    else:
        factor = None
        should_I_stop = None

    if mpi.nb_proc > 1:
        # send factor and should_I_stop
        factor = mpi.comm.bcast(factor, root=0)
        should_I_stop = mpi.comm.bcast(should_I_stop, root=0)

    if should_I_stop:
        break

    if mpi.rank == 0:
        print("factor = ", factor)
        print("nu_8 OLD = ", sim.params.nu_8)
        print("nu_8 NEW = ", sim.params.nu_8 * factor)

    it += 1
    # Modification parameters
    params, nu_8_old = modify_factor(sim)

    # Create new object simulation
    rot_fft, b_fft = get_state_from_sim(sim)
    sim = Simul(params)
    sim.state.init_statespect_from(rot_fft=rot_fft, b_fft=b_fft)
    sim.state.statephys_from_statespect()
    sim.time_stepping.start()

    if mpi.rank == 0:
        # Add values to time array and energy array
        time_total += sim.time_stepping.t
        energy, t, P_tot = compute_energy_spatial(sim)
        energy = np.mean(energy[len(energy) // 2 :])
        energies.append(energy)
        viscosities.append(sim.params.nu_8)

        # Computes new index k_max_dissipation
        kxE, kyE, dissE_kx, dissE_ky = load_mean_spect_energy_budg(
            sim, tmin=2.0, tmax=1000
        )
        idx_diss_max, idy_diss_max = _compute_ikdiss(dissE_kx, dissE_ky)
        idx_dealiasing, idy_dealiasing = _compute_ikmax(kxE, kyE)

        # Write results into temporal file
        to_print = (
            "####\n"
            "t = {:.4e} \n"
            "E = {:.4e} \n"
            "nu8 = {:.4e} \n"
            "factor = {:.4e} \n"
        ).format(time_total, energy, sim.params.nu_8, factor)

        write_to_file(path_file_write2, to_print, mode="a")

    if PLOT_FIGURES:
        ax.plot(
            kyE,
            dissE_ky,
            label="nu8 = {:.2e}, diff = {}".format(
                params.nu_8, abs(idy_diss_max - idy_dealiasing)
            ),
        )
        ax2.plot(
            kxE,
            dissE_kx,
            label="nu8 = {:.2e}, diff = {}".format(
                params.nu_8, abs(idx_diss_max - idx_dealiasing)
            ),
        )

        fig.canvas.draw()
        fig2.canvas.draw()

        plt.pause(1e-4)

        ax3.plot(time_total, viscosities[-1], "x")
        ax3.autoscale()
        fig3.canvas.draw()

        ax4.plot(time_total, energy, "x")
        ax4.autoscale()
        fig4.canvas.draw()

        ax5.plot(time_total, factor, "x")
        ax5.autoscale()
        fig5.canvas.draw()

        plt.pause(1e-4)

if mpi.rank == 0:
    if len(paths_sim) == 0:
        to_print = "gamma,nx,nu8 \n"
        to_print += "{},{},{} \n".format(gamma, sim.params.oper.nx, params.nu_8)
        mode_write = "w"

    else:
        to_print = "{},{},{} \n".format(gamma, sim.params.oper.nx, params.nu_8)
        mode_write = "a"

    write_to_file(path_file_write, to_print, mode=mode_write)

    shutil.move(sim.params.path_run, path)
