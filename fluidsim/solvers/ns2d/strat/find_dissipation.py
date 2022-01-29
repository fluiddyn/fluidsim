"""
find_dissipation (:mod:`fluidsim.solvers.ns2d.strat.find_dissipation`)
==========================================================================

To execute:

python fluidsim/solvers/ns2d/strat/find_dissipation.py 240 0.5 "nu_8"
"""

import os
import h5py
import shutil
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from math import pi
from glob import glob
from copy import deepcopy as _deepcopy

from fluidsim import load_state_phys_file
from fluidsim.solvers.ns2d.strat.solver import Simul
from fluiddyn.util import mpi


def make_parameters_simulation(gamma, key_viscosity):
    """Make parameters of the first simulation."""
    # Parameters simulation
    F = np.sin(pi / 4)
    sigma = 1

    # Create parameters
    params = Simul.create_default_params()

    # Operator parameters
    params.oper.nx = nx
    params.oper.ny = nz = nx // 4
    params.oper.Lx = Lx = 2 * pi
    params.oper.Ly = Lz = Lx * (nz / nx)
    params.oper.NO_SHEAR_MODES = True
    params.oper.coef_dealiasing = 0.6666

    # Forcing parameters
    params.forcing.enable = True
    params.forcing.type = "tcrandom_anisotropic"
    params.forcing.key_forced = "ap_fft"
    params.forcing.nkmax_forcing = nkmax_forcing = 8
    params.forcing.nkmin_forcing = nkmin_forcing = 4

    params.forcing.normalized.constant_rate_of = "energy"

    params.forcing.tcrandom_anisotropic.angle = np.arcsin(F)

    # Compute other parameters (Normalization by the energy..)
    tau_af = 1  # Forcing time equal to 1
    k_f = ((nkmax_forcing + nkmin_forcing) / 2) * max(2 * pi / Lx, 2 * pi / Lz)
    forcing_rate = (1 / tau_af**3) * ((2 * pi) / k_f) ** 2
    omega_af = 2 * pi / tau_af

    params.N = (gamma / F) * omega_af

    # Choose vis, key_viscosity=key_viscositycosity
    if key_viscosity == "nu_2":
        params.nu_2 = nu_2
    elif key_viscosity == "nu_8":
        params.nu_8 = nu_8
    else:
        raise ValueError(f"{key_viscosity} not known.")

    # Continuation on forcing...
    params.forcing.forcing_rate = forcing_rate
    params.forcing.tcrandom.time_correlation = sigma * (
        pi / (params.N * F)
    )  # time_correlation = wave period

    # Time stepping parameters
    params.time_stepping.USE_CFL = True
    params.time_stepping.t_end = t_end

    # Initialization parameters
    params.init_fields.type = "noise"

    modify_parameters(params)

    return params


def modify_parameters(params):
    """Function modifies default parameters."""
    # Output parameters
    params.output.HAS_TO_SAVE = True
    params.output.sub_directory = "find_diss_coef"
    params.output.periods_save.phys_fields = 0.4
    params.output.periods_save.spatial_means = 1e-1
    params.output.periods_save.spect_energy_budg = 1e-1
    params.output.periods_save.spectra = 1e-1


def compute_diff(idx_dealiasing, idy_dealiasing, dissE_kx, dissE_ky):
    """Computes the number of modes between the dissipation peak and the largest modes."""
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
        energy_f = params.forcing.forcing_rate ** (2 / 3) * (2 * pi / k_f) ** (
            2 / 3
        )

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
    """Computes index peak dissipation."""
    idx_diss_max = np.argmax(abs(dissE_kx))
    idy_diss_max = np.argmax(abs(dissE_ky))

    return idx_diss_max, idy_diss_max


def _compute_ikmax(kxE, kyE):
    """Computes index largest modes."""
    idx_dealiasing = np.argmin(abs(kxE - sim.oper.kxmax_dealiasing))
    idy_dealiasing = np.argmin(abs(kyE - sim.oper.kymax_dealiasing))

    return idx_dealiasing, idy_dealiasing


def compute_delta_ik(kxE, kyE, dissE_kx, dissE_ky):
    """Computes the difference between the dissipation peak and target."""
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


def modify_factor(sim, key_viscosity):
    """Modifies factor for viscosity."""
    params = _deepcopy(sim.params)

    params_old = sim.params
    sim_old = sim

    if key_viscosity == "nu_2":
        nu_old = params.nu_2
        params.nu_2 = params.nu_2 * factor
    elif key_viscosity == "nu_8":
        nu_old = params.nu_8
        params.nu_8 = params.nu_8 * factor
    else:
        raise ValueError

    params.init_fields.type = "in_script"
    params.time_stepping.t_end = t_end

    return params, nu_old


def write_to_file(path, to_print, mode="a"):
    """Writes to a file."""
    with open(path, mode) as file:
        file.write(to_print)


def make_float_value_for_path(value):
    """Makes float."""
    value_not = str(value)
    if "." in value_not:
        return value_not.split(".")[0] + "_" + value_not.split(".")[1]
    else:
        return value_not


def check_dissipation():
    """Checks dissipation"""
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

            factor = max((1 - (delta_ikmin / norm)) ** 1.5, 0.85)
            # factor = max((1 - (delta_ikmin / norm)) ** 1.5, 0.5)
            print("factor", factor)
            should_I_stop = False

        else:
            print(f"Checking stationarity... with {key_viscosity} = {nu_old}")
            E, t, P_tot = compute_energy_spatial(sim)
            ratio = np.mean(np.diff(E[2:]) / np.diff(t[2:]))

            if key_viscosity == "nu_2":
                condition_visc = abs(nu_old - params.nu_2) / params.nu_2 < 0.05
            elif key_viscosity == "nu_8":
                condition_visc = abs(nu_old - params.nu_8) / params.nu_8 < 0.05
            else:
                raise ValueError

            if (ratio / injection_energy_0) < 0.5 and condition_visc:
                if key_viscosity == "nu_2":
                    print(f"Stationarity is reached.\n nu = {params.nu_2}")
                elif key_viscosity == "nu_8":
                    print(f"Stationarity is reached.\n nu = {params.nu_8}")
                else:
                    raise ValueError

                factor = 1.0
                should_I_stop = True
            else:
                should_I_stop = False
                factor = 1.0
    else:

        factor = 1 + (1 / min(ratio_x, ratio_y))
        should_I_stop = False

    return factor, should_I_stop


if __name__ == "__main__":

    # SIMULATION
    nx = 240
    gamma = 0.5
    key_viscosity = "nu_8"
    nu_2 = 1e-3
    nu_8 = 1e-16
    t_end = 8.0
    PLOT_FIGURES = True

    # CONDITIONS
    threshold_ratio = 1e1
    min_factor = 0.7
    if nx == 240:
        nb_wavenumbers_y = 6
    elif nx == 480:
        nb_wavenumbers_y = 8
    elif nx == 960:
        nb_wavenumbers_y = 10
    else:
        raise ValueError

    gamma_not = make_float_value_for_path(gamma)

    # Create directory in path
    name_dir = "Coef_Diss_feb19"

    if key_viscosity == "nu_2":
        name_dir += "_nu2"
    elif key_viscosity == "nu_8":
        name_dir += "_nu8"
    else:
        raise ValueError

    path_root = Path("/fsnet/project/meige/2015/15DELDUCA/DataSim/")
    path_dir = path_root / name_dir
    directory_gamma = f"Coef_Diss_gamma{gamma_not}"
    path = path_dir / directory_gamma

    if mpi.rank == 0 and not path.exists():
        os.mkdir(path)

    # Check list simulations in directory
    paths_sim = sorted(path.glob("NS*"))

    # Write in .txt file
    path_file_write = path / "results.txt"

    PLOT_FIGURES = PLOT_FIGURES and mpi.rank == 0

    # Write temporal results in .txt file
    path_file_write2 = path / "results_check_nu.txt"
    if mpi.rank == 0:
        if path_file_write2.exists():
            os.remove(path_file_write2)

    if len(paths_sim) == 0:
        # Make FIRST SIMULATION
        params = make_parameters_simulation(gamma, key_viscosity)
        sim = Simul(params)
        sim = normalization_initialized_field(sim)
        sim.time_stepping.start()
    else:
        # Look for path largest resolution
        new_file = sorted(paths_sim[-1].glob("State_phys*"))[-1]
        path_file = sorted(new_file.glob("state_phys*"))[0].as_posix()

        # Compute resolution from path
        res_str = new_file.name.split("_")[-1]

        # sim = load_state_phys_file(paths_sim[-1])
        sim = load_state_phys_file(paths_sim[-1])
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
    nb_wavenumbers_x = nb_wavenumbers_y * (
        sim.params.oper.nx // sim.params.oper.ny
    )

    # Creation time and energy array
    time_total = 0
    time_total += sim.time_stepping.t

    energy, t, P_tot = compute_energy_spatial(sim)
    energy = np.mean(energy[len(energy) // 2 :])
    injection_energy_0 = P_tot[2]

    energies = []
    viscosities = []
    energies.append(energy)
    if key_viscosity == "nu_2":
        pnu = sim.params.nu_2
    elif key_viscosity == "nu_8":
        pnu = sim.params.nu_8
    else:
        raise ValueError
    viscosities.append(pnu)

    # Write results into temporal file
    if mpi.rank == 0:
        to_print = (
            "####\n"
            "t = {:.4e} \n"
            "E = {:.4e} \n"
            "nu2 = {:.4e} \n"
            "factor = {:.4e} \n"
        ).format(time_total, energy, pnu, 1)

        write_to_file(path_file_write2, to_print, mode="w")

    # Compute the data spectra energy budget
    kxE, kyE, dissE_kx, dissE_ky = sim.output.spect_energy_budg.load_mean(
        tmin=2.0, tmax=None
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
            label="nu = {:.2e}, diff = {}".format(
                sim.params.nu_2, abs(idy_diss_max - idy_dealiasing)
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
            label="nu = {:.2e}, diff = {}".format(
                sim.params.nu_2, abs(idx_diss_max - idx_dealiasing)
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
        ax3.set_ylabel(r"$\nu_2$")

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
            print("nu_2 OLD = ", sim.params.nu_2)
            print("nu_2 NEW = ", sim.params.nu_2 * factor)

        it += 1
        # Modification parameters
        params, nu_old = modify_factor(sim, key_viscosity)

        # Create new object simulation
        b_fft = sim.state.get_var("b_fft")
        rot_fft = sim.state.get_var("rot_fft")

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
            viscosities.append(sim.params.nu_2)

            # Computes new index k_max_dissipation
            kxE, kyE, dissE_kx, dissE_ky = sim.output.spect_energy_budg.load_mean(
                tmin=2.0, tmax=None
            )
            idx_diss_max, idy_diss_max = _compute_ikdiss(dissE_kx, dissE_ky)
            idx_dealiasing, idy_dealiasing = _compute_ikmax(kxE, kyE)

            # Write results into temporal file
            to_print = (
                "####\n"
                "t = {:.4e} \n"
                "E = {:.4e} \n"
                "nu = {:.4e} \n"
                "factor = {:.4e} \n"
            ).format(time_total, energy, sim.params.nu_2, factor)

            write_to_file(path_file_write2, to_print, mode="a")

        if PLOT_FIGURES:
            ax.plot(
                kyE,
                dissE_ky,
                label="nu = {:.2e}, diff = {}".format(
                    params.nu_2, abs(idy_diss_max - idy_dealiasing)
                ),
            )
            ax2.plot(
                kxE,
                dissE_kx,
                label="nu = {:.2e}, diff = {}".format(
                    params.nu_2, abs(idx_diss_max - idx_dealiasing)
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
            to_print = "gamma,nx,nu \n"
            to_print += "{},{},{} \n".format(
                gamma, sim.params.oper.nx, params.nu_2
            )
            mode_write = "w"

        else:
            to_print = "{},{},{} \n".format(
                gamma, sim.params.oper.nx, params.nu_2
            )
            mode_write = "a"

        write_to_file(path_file_write, to_print, mode=mode_write)

        shutil.move(sim.params.path_run, path)
