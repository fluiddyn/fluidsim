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

from __future__ import print_function

import os
import h5py
import shutil
import numpy as np
import matplotlib.pyplot as plt

from math import pi
from glob import glob
from copy import deepcopy as _deepcopy

from fluidsim import load_sim_for_plot
from fluidsim.solvers.ns2d.strat.solver import Simul
from fluiddyn.util import mpi

from ns2dstrat_lmode import make_parameters_simulation, modify_parameters

def load_mean_spect_energy_budg(sim, tmin=0, tmax=1000):
    """
    Loads data spect_energy_budget.
    It computes the mean between tmin and tmax.
    """
    print("path_file", sim.output.spect_energy_budg.path_file)
    with h5py.File(sim.output.spect_energy_budg.path_file, "r") as f:
        times = f["times"].value
        kxE = f["kxE"].value
        kyE = f["kyE"].value
        dset_dissEK_kx = f["dissEK_kx"].value
        dset_dissEA_kx = f["dissEA_kx"].value
        dset_dissEK_ky = f["dissEK_ky"].value
        dset_dissEA_ky = f["dissEA_ky"].value

        imin_plot = np.argmin(abs(times - tmin))
        imax_plot = np.argmin(abs(times - tmax))

        dset_dissE_kx = dset_dissEK_kx + dset_dissEA_kx
        dissE_kx = dset_dissE_kx[imin_plot:imax_plot + 1].mean(0)

        dset_dissE_ky = dset_dissEK_ky + dset_dissEA_ky
        dissE_ky = dset_dissE_ky[imin_plot:imax_plot + 1].mean(0)

    return kxE, kyE, dissE_kx, dissE_ky

def get_state_from_sim(sim):
    """Returns the state from a simulation."""
    # Take the state
    b_fft = sim.state.get_var('b_fft')
    rot_fft = sim.state.get_var('rot_fft')

    return rot_fft, b_fft

def compute_diff(idx_dealiasing, idy_dealiasing, dissE_kx, dissE_ky):
    idx_diss_max = np.argmax(abs(dissE_kx))
    idy_diss_max = np.argmax(abs(dissE_ky))

    # Computes difference
    diff_x = idx_dealiasing -  idx_diss_max
    diff_y = idy_dealiasing -  idy_diss_max

    diff = np.argmax([diff_x, diff_y])

    if diff == 0:
        print("diff_x = idx_dealiasing -  idx_diss_max", idx_dealiasing -  idx_diss_max)
        diff = idx_dealiasing -  idx_diss_max

    else:
        print("diff_y = idy_dealiasing -  idy_diss_max", idy_dealiasing -  idy_diss_max)
        diff = idy_dealiasing -  idy_diss_max

    return diff, idx_diss_max, idy_diss_max

def normalization_initialized_field(sim, coef_norm=1e-4):
    """Normalizes the initialized field. (ONLY if nx != ny)"""
    if sim.params.oper.nx != sim.params.oper.ny:

        if not sim.params.forcing.key_forced == "ap_fft":
            raise ValueError("sim.params.forcing.key_forced should be ap_fft.")

        KX = sim.oper.KX
        cond = KX == 0.

        ux_fft = sim.state.get_var('ux_fft')
        uy_fft = sim.state.get_var('uy_fft')
        b_fft = sim.state.get_var('b_fft')

        ux_fft[cond] = 0.
        uy_fft[cond] = 0.
        b_fft[cond] = 0.

        # Compute energy after ux_fft[kx=0] uy_fft[kx=0] b_fft[kx=0]
        ek_fft = (np.abs(ux_fft)**2 + np.abs(uy_fft)**2)/2
        ea_fft = ((np.abs(b_fft)/params.N)**2)/2
        e_fft = ek_fft + ea_fft
        energy_before_norm = sim.output.sum_wavenumbers(e_fft)

        # Compute scale energy forcing
        Lx = sim.params.oper.Lx
        Lz = sim.params.oper.Ly

        nkmax_forcing = params.forcing.nkmax_forcing
        nkmin_forcing = params.forcing.nkmin_forcing

        k_f = ((nkmax_forcing + nkmin_forcing) / 2) * max(2 * pi / Lx, 2 * pi / Lz)
        energy_f = params.forcing.forcing_rate**(2/7) * (2 * pi / k_f)**7

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



#################################################
### Parameters script ###
# Parameters simulations
gamma = 1.
F = np.sin(pi / 4) # F = omega_l / N
sigma = 1 # sigma = omega_l / (pi * f_cf); f_cf freq time correlation forcing in s-1
nu_8 = 1e-16

coef_modif_resol = 3 / 2

NO_SHEAR_MODES = False
min_factor = 0.7

# Notation for gamma
gamma_not = str(gamma)
if "." in gamma_not:
    gamma_not = gamma_not.split(".")[0] + "_" + gamma_not.split(".")[1]

# Create directory in DataSim
path_root_dir = "/fsnet/project/meige/2015/15DELDUCA/DataSim"
path_dir = os.path.join(path_root_dir, "Coef_Diss_gamma{}".format(gamma_not))
if mpi.rank == 0 and  not os.path.exists(path_dir):
    os.mkdir(path_dir)

# Check list simulations in directory
paths_sim = sorted(glob(os.path.join(path_dir, "NS*")))

# Write in .txt file
path_file = os.path.join(path_dir, "results.txt")

# import sys
# sys.exit()

# paths_sim = []
# resolutions = []
# dissipations = []

PLOT_FIGURES = True
PLOT_FIGURES = PLOT_FIGURES and mpi.rank == 0

if len(paths_sim) == 0:

    params =  make_parameters_simulation(gamma, F, sigma, nu_8, t_end=8., NO_SHEAR_MODES=NO_SHEAR_MODES)
    sim = Simul(params)

    # Normalization of the field and start
    sim = normalization_initialized_field(sim)

    sim.time_stepping.start()

    # Parameters condition
    nb_wavenumbers_y = 8
    nb_wavenumbers_x = nb_wavenumbers_y * (sim.params.oper.nx // sim.params.oper.ny)

    # Creation time and energy array
    time_total = 0
    energies = []
    viscosities = []

    time_total += sim.time_stepping.t

    dict_spatial = sim.output.spatial_means.load()
    energy = dict_spatial["E"]
    energy = np.mean(energy[len(energy) // 2:])
    energies.append(energy)
    viscosities.append(sim.params.nu_8)

    # Compute injection of energy begin simulation
    pe_k = dict_spatial["PK_tot"]
    pe_a = dict_spatial["PA_tot"]
    pe_tot = pe_k + pe_a
    print("pe_tot", pe_tot[2])

    injection_energy_0 = pe_tot[2]

    # Compute the data spectra energy budget
    kxE, kyE, dissE_kx, dissE_ky = load_mean_spect_energy_budg(sim, tmin=2., tmax=1000)

    # Loads the spectral energy budget
    kxmax_dealiasing = sim.oper.kxmax_dealiasing
    kymax_dealiasing = sim.oper.kymax_dealiasing

    # Computes index kmax_dealiasing & kmax_dissipation
    idx_dealiasing = np.argmin(abs(kxE - kxmax_dealiasing))
    idy_dealiasing = np.argmin(abs(kyE - kymax_dealiasing))

    # Compute difference
    diff, idx_diss_max, idy_diss_max = compute_diff(idx_dealiasing, idy_dealiasing, dissE_kx, dissE_ky)

    #
    idx_target = idx_dealiasing - (nb_wavenumbers_x - 1)
    idy_target = idy_dealiasing - (nb_wavenumbers_y - 1)


    # Plot dissipation
    if PLOT_FIGURES:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("$D(k_y)$")
        ax.set_xlabel("$k_y$")
        ax.set_ylabel("$D(k_y)$")
        ax.plot(kyE, dissE_ky, label="nu8 = {:.2e}, diff = {}".format(
            sim.params.nu_8, abs(idy_diss_max - idy_dealiasing)))
        ax.plot(kymax_dealiasing, 0, 'xr')
        ax.axvline(x=sim.oper.deltaky * idy_target, color="k")

        fig2, ax2 = plt.subplots()
        ax2.set_title("$D(k_x)$")
        ax2.set_xlabel("$k_x$")
        ax2.set_ylabel("$D(k_x)$")
        ax2.plot(kxE, dissE_kx, label="nu8 = {:.2e}, diff = {}".format(
            sim.params.nu_8, abs(idx_diss_max - idx_dealiasing)))
        ax2.plot(kxmax_dealiasing, 0, 'xr')
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

        ax3.plot(time_total, viscosities[-1], '.')

        # Energy Vs time
        fig4, ax4 = plt.subplots()
        ax4.plot(time_total, energy, '.')
        ax4.set_xlabel("times")
        ax4.set_ylabel("Energy")

        # Factor Vs time
        fig5, ax5 = plt.subplots()
        ax5.plot(time_total, 1, '.')
        ax5.set_xlabel("times")
        ax5.set_ylabel("Factor")


    it = 0
    p = 1
    # Check ...
    while True:

        if mpi.rank == 0:
            # Define conditions
            diff_x = abs(idx_dealiasing - idx_diss_max)
            diff_y = abs(idy_dealiasing - idy_diss_max)
            print("diff_x", diff_x)
            print("diff_y", diff_y)

            ratio_x = dissE_kx[idx_diss_max] / dissE_kx[idx_dealiasing - 1]
            ratio_y = dissE_ky[idy_diss_max] / dissE_ky[idy_dealiasing - 1]

            cond_ratio_x = ratio_x > 1e1
            cond_ratio_y = ratio_y > 1e1
            print("cond_ratio_x", ratio_x)
            print("cond_ratio_y", ratio_y)

            diff_x_target = abs(idx_target - idx_diss_max)
            diff_y_target = abs(idy_target - idy_diss_max)
            diff_target = max(diff_x_target, diff_y_target)

            if time_total > 1000:
                print(
                    "The stationarity has not " + \
                    "reached after {} simulations.".format(it_))
                break
            # Check ratio D(k_peak) / D(k_max - 1)
            if cond_ratio_x and cond_ratio_y:

                # Check differences
                if diff_x > nb_wavenumbers_x and diff_y > nb_wavenumbers_y:
                    print("diff_target = ", diff_target)
                    print("p", p)
                    factor = max(((nb_wavenumbers_y  / 2) / diff_target) ** (0.2), min_factor)
                    print("factor = ", factor)

                    p += 1
                    should_I_stop = False
                else:
                    print("Checking stationarity... with nu8 = {}".format(params_old.nu_8))
                    dict_spatial = sim.output.spatial_means.load()
                    E = dict_spatial["E"]
                    t = dict_spatial["t"]
                    ratio = np.mean(np.diff(E[2:]) / np.diff(t[2:]))
                    print("ratio_energy = ", ratio)
                    print("injection_energy_0 = ", injection_energy_0)

                    print("nu_8_old", nu_8_old)
                    print("nu_8", params.nu_8)
                    print("abs(nu_8_old - nu_8) = ", abs(nu_8_old - params.nu_8))
                    print("abs(nu_8_old - nu_8) / nu_8 = ", abs(nu_8_old - params.nu_8) / params.nu_8)
                    if (ratio / injection_energy_0) < 0.5 and \
                       abs(nu_8_old - params.nu_8) / params.nu_8 < 0.05:

                        print(f"Stationarity is reached.\n nu_8 = {params.nu_8}")
                        # sim.output.phys_fields.plot()
                        should_I_stop = True
                        # break
                    else:
                        should_I_stop = False

                    factor = 1.

            else:

                factor = 1 + (1 / min(ratio_x, ratio_y))
                print("factor = ", factor)
                p += 1
                should_I_stop = False

            # Print values...
            print("params.nu_8", sim.params.nu_8)
            print("abs(idx_dealiasing - idx_diss_max)", diff_x)
            print("abs(idy_dealiasing - idy_diss_max)", diff_y)
            print("cond_ratio_x", dissE_kx[idx_diss_max] / dissE_kx[idx_dealiasing - 1])
            print("cond_ratio_y", dissE_ky[idy_diss_max] / dissE_ky[idy_dealiasing - 1])
        else:
            factor = None
            should_I_stop = None

        if mpi.nb_proc > 1:
            # send factor and should_I_stop
            factor = mpi.comm.bcast(factor, root=0)
            should_I_stop = mpi.comm.bcast(should_I_stop, root=0)
            print("rank {} ; factor {}".format(mpi.comm.Get_rank(), factor))

        if should_I_stop:
           break

        it += 1
        # Modification parameters
        params = _deepcopy(sim.params)

        nu_8_old = params.nu_8
        params_old = sim.params
        sim_old = sim

        params.nu_8 = params.nu_8 * factor
        params.init_fields.type = 'in_script'
        params.time_stepping.t_end = 8.

        # Create new object simulation
        rot_fft, b_fft = get_state_from_sim(sim)
        sim = Simul(params)
        sim.state.init_statespect_from(rot_fft=rot_fft, b_fft=b_fft)
        sim.state.statephys_from_statespect()
        sim.time_stepping.start()

        # Add values to time array and energy array
        time_total += sim.time_stepping.t
        dict_spatial = sim.output.spatial_means.load()
        energy = dict_spatial["E"]
        energy = np.mean(energy[len(energy) // 2:])
        energies.append(energy)
        viscosities.append(sim.params.nu_8)

        # Computes new index k_max_dissipation
        kxE, kyE, dissE_kx, dissE_ky = load_mean_spect_energy_budg(
            sim, tmin=2, tmax=1000)

        diff, idx_diss_max, idy_diss_max = compute_diff(idx_dealiasing, idy_dealiasing, dissE_kx, dissE_ky)

        if PLOT_FIGURES:
            ax.plot(kyE, dissE_ky, label="nu8 = {:.2e}, diff = {}".format(
                params.nu_8, abs(idy_diss_max - idy_dealiasing)))
            ax2.plot(kxE, dissE_kx, label="nu8 = {:.2e}, diff = {}".format(
                params.nu_8, abs(idx_diss_max - idx_dealiasing)))

            fig.canvas.draw()
            fig2.canvas.draw()

            plt.pause(1e-4)

            ax3.plot(time_total, viscosities[-1], "x")
            ax3.autoscale()
            fig3.canvas.draw()

            ax4.plot(time_total, energy, "x")
            ax4.autoscale()
            fig4.canvas.draw()

            ax5.plot(time_total, factor, 'x')
            ax5.autoscale()
            fig5.canvas.draw()

            plt.pause(1e-4)
    if mpi.rank == 0:
        with open(path_file, "w") as f:
            to_print = ("resolution = {} \n"
                        "nu8 = {} \n".format(
                            sim.params.oper.nx,
                            params.nu_8))

            f.write(to_print)

        shutil.move(sim.params.path_run, path_dir)



else:
    plt.close("all")

    sim = load_sim_for_plot(paths_sim[-1])

    params = _deepcopy(sim.params)

    params.oper.nx = int(params.oper.nx * coef_modif_resol)
    params.oper.ny = int(params.oper.ny * coef_modif_resol)

    params.init_fields.type = "from_file"
    params.init_fields.from_file.path = paths_sim[-1] + "/State_phys_360x90/state_phys_t008.002_it=0.nc"

    params.time_stepping.t_end += 8.

    params.NEW_DIR_RESULTS = True

    modify_parameters(params)

    sim = Simul(params)
    sim.time_stepping.start()

    # Parameters condition
    nb_wavenumbers_y = 8
    nb_wavenumbers_x = nb_wavenumbers_y * (sim.params.oper.nx // sim.params.oper.ny)

    # Creation time and energy array
    time_total = 0
    time_total += sim.time_stepping.t

    energies = []
    viscosities = []

    dict_spatial = sim.output.spatial_means.load()
    energy = dict_spatial["E"]
    energy = np.mean(energy[len(energy) // 2:])
    energies.append(energy)
    viscosities.append(sim.params.nu_8)

    # Compute injection of energy begin simulation
    pe_k = dict_spatial["PK_tot"]
    pe_a = dict_spatial["PA_tot"]
    pe_tot = pe_k + pe_a
    print("pe_tot", pe_tot[2])

    injection_energy_0 = pe_tot[2]

    # Compute the data spectra energy budget
    kxE, kyE, dissE_kx, dissE_ky = load_mean_spect_energy_budg(sim, tmin=2., tmax=1000)
    print("kxE", kxE)
    # Save nu_8 sim0 :
    nu8_old = sim.params.nu_8

    # Loads the spectral energy budget
    kxmax_dealiasing = sim.oper.kxmax_dealiasing
    kymax_dealiasing = sim.oper.kymax_dealiasing
    print("kxmax_dealiasing", kxmax_dealiasing)
    # Computes index kmax_dealiasing & kmax_dissipation
    idx_dealiasing = np.argmin(abs(kxE - kxmax_dealiasing))
    idy_dealiasing = np.argmin(abs(kyE - kymax_dealiasing))

    # Compute difference
    diff, idx_diss_max, idy_diss_max = compute_diff(idx_dealiasing, idy_dealiasing, dissE_kx, dissE_ky)

    #
    idx_target = idx_dealiasing - (nb_wavenumbers_x - 1)
    idy_target = idy_dealiasing - (nb_wavenumbers_y - 1)

    if PLOT_FIGURES:
        # Plot dissipation
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("$D(k_y)$")
        ax.set_xlabel("$k_y$")
        ax.set_ylabel("$D(k_y)$")
        ax.plot(kyE, dissE_ky, label="nu8 = {:.2e}, diff = {}".format(
            sim.params.nu_8, abs(idy_diss_max - idy_dealiasing)))
        ax.plot(kymax_dealiasing, 0, 'xr')
        ax.axvline(x=sim.oper.deltaky * idy_target, color="k")

        fig2, ax2 = plt.subplots()
        ax2.set_title("$D(k_x)$")
        ax2.set_xlabel("$k_x$")
        ax2.set_ylabel("$D(k_x)$")
        ax2.plot(kxE, dissE_kx, label="nu8 = {:.2e}, diff = {}".format(
            sim.params.nu_8, abs(idx_diss_max - idx_dealiasing)))
        ax2.plot(kxmax_dealiasing, 0, 'xr')
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

        ax3.plot(time_total, viscosities[-1], '.')

        # Energy Vs time
        fig4, ax4 = plt.subplots()
        ax4.plot(time_total, energy, '.')
        ax4.set_xlabel("times")
        ax4.set_ylabel("Energy")

        # Factor Vs time
        fig5, ax5 = plt.subplots()
        ax5.plot(time_total, 1, '.')
        ax5.set_xlabel("times")
        ax5.set_ylabel("Factor")


    it = 0
    p = 1
    # Check ...
    while True:

        if mpi.rank == 0:

            # Define conditions
            diff_x = abs(idx_dealiasing - idx_diss_max)
            diff_y = abs(idy_dealiasing - idy_diss_max)
            print("diff_x", diff_x)
            print("diff_y", diff_y)

            ratio_x = dissE_kx[idx_diss_max] / dissE_kx[idx_dealiasing - 1]
            ratio_y = dissE_ky[idy_diss_max] / dissE_ky[idy_dealiasing - 1]

            cond_ratio_x = ratio_x > 1e1
            cond_ratio_y = ratio_y > 1e1
            print("cond_ratio_x", ratio_x)
            print("cond_ratio_y", ratio_y)

            diff_x_target = abs(idx_target - idx_diss_max)
            diff_y_target = abs(idy_target - idy_diss_max)
            diff_target = max(diff_x_target, diff_y_target)

            if time_total > 1000:
                print(
                    "The stationarity has not " + \
                    "reached after {} simulations.".format(it))
                break
            # Check ratio D(k_peak) / D(k_max - 1)
            if cond_ratio_x and cond_ratio_y:

                # Check differences
                if diff_x > nb_wavenumbers_x and diff_y > nb_wavenumbers_y:
                    print("diff_target = ", diff_target)
                    print("p", p)
                    # factor = max(((nb_wavenumbers_y  / 2) / diff_target) ** (0.2), min_factor)
                    factor = ((nb_wavenumbers_y  / 2) / diff_target) ** (1. / p)
                    print("factor = ", factor)

                    p += 1
                    should_I_stop = False
                else:
                    print("Checking stationarity... with nu8 = {}".format(params_old.nu_8))
                    dict_spatial = sim.output.spatial_means.load()
                    E = dict_spatial["E"]
                    t = dict_spatial["t"]
                    ratio = np.mean(np.diff(E[2:]) / np.diff(t[2:]))
                    print("ratio_energy = ", ratio)
                    print("injection_energy_0 = ", injection_energy_0)

                    print("nu_8_old", nu_8_old)
                    print("nu_8", params.nu_8)
                    print("abs(nu_8_old - nu_8) = ", abs(nu_8_old - params.nu_8))
                    print("abs(nu_8_old - nu_8) / nu_8 = ", abs(nu_8_old - params.nu_8) / params.nu_8)
                    if (ratio / injection_energy_0) < 0.5 and \
                       abs(nu_8_old - params.nu_8) / params.nu_8 < 0.05:

                        print(f"Stationarity is reached.\n nu_8 = {params.nu_8}")
                        should_I_stop = True
                        # sim.output.phys_fields.plot()
                        # break
                    else:
                        should_I_stop = False

                    factor = 1.


            else:

                factor = 1 + (1 / min(ratio_x, ratio_y))
                print("factor = ", factor)
                p += 1
                should_I_stop = False

            # Print values...
            print("params.nu_8", sim.params.nu_8)
            print("abs(idx_dealiasing - idx_diss_max)", diff_x)
            print("abs(idy_dealiasing - idy_diss_max)", diff_y)
            print("cond_ratio_x", dissE_kx[idx_diss_max] / dissE_kx[idx_dealiasing - 1])
            print("cond_ratio_y", dissE_ky[idy_diss_max] / dissE_ky[idy_dealiasing - 1])

        else:
            factor = None
            should_I_stop = None

            if mpi.nb_proc > 1:
                # send factor and should_I_stop
                factor = mpi.comm.bcast(factor, root=0)
                should_I_stop = mpi.comm.bcast(should_I_stop, root=0)

        if should_I_stop:
           break

        it += 1
        # Modification parameters
        params = _deepcopy(sim.params)

        nu_8_old = params.nu_8
        params_old = sim.params
        sim_old = sim

        params.nu_8 = params.nu_8 * factor
        params.init_fields.type = 'in_script'
        params.time_stepping.t_end = 8.

        # Create new object simulation
        rot_fft, b_fft = get_state_from_sim(sim)
        sim = Simul(params)
        sim.state.init_statespect_from(rot_fft=rot_fft, b_fft=b_fft)
        sim.state.statephys_from_statespect()
        sim.time_stepping.start()

        if mpi.rank == 0:

            # Add values to time array and energy array
            time_total += sim.time_stepping.t
            dict_spatial = sim.output.spatial_means.load()
            energy = dict_spatial["E"]
            energy = np.mean(energy[len(energy) // 2:])
            energies.append(energy)
            viscosities.append(sim.params.nu_8)

            # Computes new index k_max_dissipation
            kxE, kyE, dissE_kx, dissE_ky = load_mean_spect_energy_budg(
                sim, tmin=2, tmax=1000)

            diff, idx_diss_max, idy_diss_max = compute_diff(idx_dealiasing, idy_dealiasing, dissE_kx, dissE_ky)

        if PLOT_FIGURES:

            ax.plot(kyE, dissE_ky, label="nu8 = {:.2e}, diff = {}".format(
                params.nu_8, abs(idy_diss_max - idy_dealiasing)))
            ax2.plot(kxE, dissE_kx, label="nu8 = {:.2e}, diff = {}".format(
                params.nu_8, abs(idx_diss_max - idx_dealiasing)))

            fig.canvas.draw()
            fig2.canvas.draw()

            plt.pause(1e-4)

            ax3.plot(time_total, viscosities[-1], "x")
            ax3.autoscale()
            fig3.canvas.draw()

            ax4.plot(time_total, energy, "x")
            ax4.autoscale()
            fig4.canvas.draw()

            ax5.plot(time_total, factor, 'x')
            ax5.autoscale()
            fig5.canvas.draw()

            plt.pause(1e-4)
    if mpi.rank == 0:
        with open(path_file, "r+") as f:
            to_print = ("resolution = {} \n"
                        "nu8 = {} \n".format(
                            sim.params.oper.nx,
                            params.nu_8))

            f.write(to_print)

        shutil.move(sim.params.path_run, path_dir)

    # modif_resolution (in util)
    # pass




##### SAVE #####
# nb_wavenumbers_y = 8
# nb_wavenumbers_x = nb_wavenumbers_y * (params.oper.nx // params.oper.ny)

# # Creation time and energy array
# time_total = 0
# energies = []
# viscosities = []

# sim.time_stepping.start()

# time_total += sim.time_stepping.t

# dict_spatial = sim.output.spatial_means.load()
# energy = dict_spatial["E"]
# energy = np.mean(energy[len(energy) // 2:])
# energies.append(energy)
# viscosities.append(sim.params.nu_8)

# # Compute injection of energy begin simulation
# pe_k = dict_spatial["PK_tot"]
# pe_a = dict_spatial["PA_tot"]
# pe_tot = pe_k + pe_a
# print("pe_tot", pe_tot[2])

# injection_energy_0 = pe_tot[2]

# # Compute the data spectra energy budget
# kxE, kyE, dissE_kx, dissE_ky = load_mean_spect_energy_budg(sim, tmin=2., tmax=1000)

# # Save nu_8 sim0 :
# nu8_old = sim.params.nu_8

# # Loads the spectral energy budget
# kxmax_dealiasing = sim.oper.kxmax_dealiasing
# kymax_dealiasing = sim.oper.kymax_dealiasing

# # Computes index kmax_dealiasing & kmax_dissipation
# idx_dealiasing = np.argmin(abs(kxE - kxmax_dealiasing))
# idy_dealiasing = np.argmin(abs(kyE - kymax_dealiasing))

# # Compute difference
# diff, idx_diss_max, idy_diss_max = compute_diff(idx_dealiasing, idy_dealiasing, dissE_kx, dissE_ky)

# #
# idx_target = idx_dealiasing - (nb_wavenumbers_x - 1)
# idy_target = idy_dealiasing - (nb_wavenumbers_y - 1)


# # Plot dissipation
# plt.ion()
# fig, ax = plt.subplots()
# ax.set_title("$D(k_y)$")
# ax.set_xlabel("$k_y$")
# ax.set_ylabel("$D(k_y)$")
# ax.plot(kyE, dissE_ky, label="nu8 = {:.2e}, diff = {}".format(
#     sim.params.nu_8, abs(idy_diss_max - idy_dealiasing)))
# ax.plot(kymax_dealiasing, 0, 'xr')
# ax.axvline(x=sim.oper.ky[idy_target], color="k")

# fig2, ax2 = plt.subplots()
# ax2.set_title("$D(k_x)$")
# ax2.set_xlabel("$k_x$")
# ax2.set_ylabel("$D(k_x)$")
# ax2.plot(kxE, dissE_kx, label="nu8 = {:.2e}, diff = {}".format(
#     sim.params.nu_8, abs(idx_diss_max - idx_dealiasing)))
# ax2.plot(kxmax_dealiasing, 0, 'xr')
# ax2.axvline(x=sim.oper.kx[idx_target], color="k")


# ax.legend()
# ax2.legend()

# fig.canvas.draw()
# fig2.canvas.draw()

# plt.pause(1e-3)


# # Dissipation vs time
# fig3, ax3 = plt.subplots()
# ax3.set_xlabel("times")
# ax3.set_ylabel(r"$\nu_8$")

# ax3.plot(time_total, viscosities[-1], '.')

# # Energy Vs time
# fig4, ax4 = plt.subplots()
# ax4.plot(time_total, energy, '.')
# ax4.set_xlabel("times")
# ax4.set_ylabel("Energy")

# # Factor Vs time
# fig5, ax5 = plt.subplots()
# ax5.plot(time_total, 1, '.')
# ax5.set_xlabel("times")
# ax5.set_ylabel("Factor")


# it = 0
# p = 1
# # Check ...
# while True:

#     # Define conditions
#     diff_x = abs(idx_dealiasing - idx_diss_max)
#     diff_y = abs(idy_dealiasing - idy_diss_max)
#     print("diff_x", diff_x)
#     print("diff_y", diff_y)

#     ratio_x = dissE_kx[idx_diss_max] / dissE_kx[idx_dealiasing - 1]
#     ratio_y = dissE_ky[idy_diss_max] / dissE_ky[idy_dealiasing - 1]

#     cond_ratio_x = ratio_x > 1e1
#     cond_ratio_y = ratio_y > 1e1
#     print("cond_ratio_x", ratio_x)
#     print("cond_ratio_y", ratio_y)

#     diff_x_target = abs(idx_target - idx_diss_max)
#     diff_y_target = abs(idy_target - idy_diss_max)
#     diff_target = max(diff_x_target, diff_y_target)

#     if time_total > 1000:
#         print(
#             "The stationarity has not " + \
#             "reached after {} simulations.".format(it_))
#         break

#     # Check ratio D(k_peak) / D(k_max - 1)
#     if cond_ratio_x and cond_ratio_y:

#         # Check differences
#         if diff_x > nb_wavenumbers_x and diff_y > nb_wavenumbers_y:
#             print("diff_target = ", diff_target)
#             print("p", p)
#             factor = ((nb_wavenumbers_y  / 2) / diff_target) ** (0.1 / p)
#             print("factor = ", factor)

#             p += 1
#         else:
#             print("Checking stationarity... with nu8 = {}".format(params_old.nu_8))
#             dict_spatial = sim.output.spatial_means.load()
#             E = dict_spatial["E"]
#             t = dict_spatial["t"]
#             ratio = np.mean(np.diff(E[2:]) / np.diff(t[2:]))
#             print("ratio_energy = ", ratio)
#             print("injection_energy_0 = ", injection_energy_0)

#             print("nu_8_old", nu_8_old)
#             print("nu_8", params.nu_8)
#             print("abs(nu_8_old - nu_8) = ", abs(nu_8_old - params.nu_8))
#             print("abs(nu_8_old - nu_8) / nu_8 = ", abs(nu_8_old - params.nu_8) / params.nu_8)
#             if (ratio / injection_energy_0) < 0.5 and \
#                abs(nu_8_old - params.nu_8) / params.nu_8 < 0.05:

#                 print(f"Stationarity is reached.\n nu_8 = {params.nu_8}")
#                 sim.output.phys_fields.plot()
#                 break

#             factor = 1.

#     else:

#         factor = 1 + (1 / min(ratio_x, ratio_y))
#         print("factor = ", factor)
#         p += 1

#     # Print values...
#     print("params.nu_8", sim.params.nu_8)
#     print("abs(idx_dealiasing - idx_diss_max)", diff_x)
#     print("abs(idy_dealiasing - idy_diss_max)", diff_y)
#     print("cond_ratio_x", dissE_kx[idx_diss_max] / dissE_kx[idx_dealiasing - 1])
#     print("cond_ratio_y", dissE_ky[idy_diss_max] / dissE_ky[idy_dealiasing - 1])

#     it += 1
#     # Modification parameters
#     params = _deepcopy(sim.params)

#     nu_8_old = params.nu_8
#     params_old = sim.params
#     sim_old = sim

#     params.nu_8 = params.nu_8 * factor
#     params.init_fields.type = 'in_script'
#     params.time_stepping.t_end = 8.

#     # Create new object simulation
#     rot_fft, b_fft = get_state_from_sim(sim)
#     sim = Simul(params)
#     sim.state.init_statespect_from(rot_fft=rot_fft, b_fft=b_fft)
#     sim.state.statephys_from_statespect()
#     sim.time_stepping.start()

#     # Add values to time array and energy array
#     time_total += sim.time_stepping.t
#     dict_spatial = sim.output.spatial_means.load()
#     energy = dict_spatial["E"]
#     energy = np.mean(energy[len(energy) // 2:])
#     energies.append(energy)
#     viscosities.append(sim.params.nu_8)


#     # Computes new index k_max_dissipation
#     kxE, kyE, dissE_kx, dissE_ky = load_mean_spect_energy_budg(
#         sim, tmin=2, tmax=1000)

#     diff, idx_diss_max, idy_diss_max = compute_diff(idx_dealiasing, idy_dealiasing, dissE_kx, dissE_ky)

#     ax.plot(kyE, dissE_ky, label="nu8 = {:.2e}, diff = {}".format(
#         params.nu_8, abs(idy_diss_max - idy_dealiasing)))
#     ax2.plot(kxE, dissE_kx, label="nu8 = {:.2e}, diff = {}".format(
#         params.nu_8, abs(idx_diss_max - idx_dealiasing)))

#     # ax.legend()
#     # ax2.legend()

#     fig.canvas.draw()
#     fig2.canvas.draw()

#     plt.pause(1e-4)

#     # its.append(it)
#     # viscosities.append(params.nu_8)
#     # line.set_data(its, viscosities)

#     ax3.plot(time_total, viscosities[-1], "x")
#     ax3.autoscale()
#     fig3.canvas.draw()

#     ax4.plot(time_total, energy, "x")
#     ax4.autoscale()
#     fig4.canvas.draw()

#     ax5.plot(time_total, factor, 'x')
#     ax5.autoscale()
#     fig5.canvas.draw()

#     plt.pause(1e-4)
