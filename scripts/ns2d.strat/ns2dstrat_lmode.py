"""
ns2dstrat_lmode.py
==================

"""

import numpy as np

from math import pi
from fluidsim.solvers.ns2d.strat.solver import Simul


def make_parameters_simulation(
    gamma, F, sigma, nu_8, t_end=10, NO_SHEAR_MODES=False
):

    ## Operator parameters
    anisotropy_domain = 4  # anisotropy_domain = nx / nz
    nx = 240
    nz = nx // anisotropy_domain
    Lx = 2 * pi
    Lz = Lx * (nz / nx)  # deltax = deltay

    coef_dealiasing = 0.6666

    # Time stepping parameters
    USE_CFL = True
    deltat0 = 0.0005
    # t_end = 5.

    ## Forcing parameters
    forcing_enable = True
    nkmax_forcing = 8
    nkmin_forcing = 4
    tau_af = 1  # Forcing time equal to 1

    ######################
    #######################

    # Create parameters
    params = Simul.create_default_params()

    # Operator parameters
    params.oper.nx = nx
    params.oper.ny = nz
    params.oper.Lx = Lx
    params.oper.Ly = Lz
    params.oper.NO_SHEAR_MODES = NO_SHEAR_MODES
    params.oper.coef_dealiasing = coef_dealiasing

    # Forcing parameters
    params.forcing.enable = forcing_enable
    params.forcing.type = "tcrandom_anisotropic"
    params.forcing.key_forced = "ap_fft"
    params.forcing.nkmax_forcing = nkmax_forcing
    params.forcing.nkmin_forcing = nkmin_forcing

    params.forcing.tcrandom_anisotropic.angle = np.arcsin(F)

    # Compute other parameters
    k_f = ((nkmax_forcing + nkmin_forcing) / 2) * max(2 * pi / Lx, 2 * pi / Lz)
    forcing_rate = (1 / tau_af**7) * ((2 * pi) / k_f) ** 2
    omega_af = 2 * pi / tau_af
    params.N = (gamma / F) * omega_af
    params.nu_8 = nu_8

    # Continuation on forcing...
    params.forcing.forcing_rate = forcing_rate
    params.forcing.tcrandom.time_correlation = sigma * (
        pi / (params.N * F)
    )  # time_correlation = wave period

    # Time stepping parameters
    params.time_stepping.USE_CFL = USE_CFL
    params.time_stepping.deltat0 = deltat0
    params.time_stepping.t_end = t_end

    # Initialization parameters
    params.init_fields.type = "noise"

    modify_parameters(params)

    return params


def modify_parameters(params):

    # Output parameters
    params.output.HAS_TO_SAVE = True
    params.output.sub_directory = "find_diss_coef"
    params.output.periods_save.spatial_means = 1e-1
    params.output.periods_save.spect_energy_budg = 1e-1
    params.output.periods_save.spectra = 1e-1


if __name__ == "__main__":

    ##### PARAMETERS #####
    ######################
    gamma = 0.2  # gamma = omega_l / omega_af
    F = np.sin(pi / 4)  # F = omega_l / N
    sigma = 1  # sigma = omega_l / (pi * f_cf); f_cf freq time correlation forcing in s-1
    nu_8 = 1e-15

    params = make_parameters_simulation(gamma, F, sigma, nu_8, t_end=50.0)

    # Start simulation
    sim = Simul(params)

    #####################################
    # START NORMALIZATION INITIALIZATION (if nx != nz)
    # Normalize initialization
    if sim.params.oper.nx != sim.params.oper.ny:
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

        coef = np.sqrt(1e-4 * energy_f / energy_before_norm)

        ux_fft *= coef
        uy_fft *= coef
        b_fft *= coef

        rot_fft = sim.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        sim.state.init_statespect_from(rot_fft=rot_fft, b_fft=b_fft)

        sim.state.statephys_from_statespect()
        # END NORMALIZATION INITIALIZATION
        ###################################

    sim.time_stepping.start()
