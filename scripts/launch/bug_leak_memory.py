# python bug_leak_memory.py

from __future__ import print_function

from fluidsim.solvers.ns2d.strat.solver import Simul
from math import pi, degrees

import numpy as np


def _make_params_sim(R, F):
    """
    Make parameters of the simulation.
    """
    params = Simul.create_default_params()
    params.oper.nx = nh = 48
    params.oper.ny = nh
    # params.oper.Lx = params.oper.Ly = Lh = 2 * pi
    params.oper.coef_dealiasing = 0.5

    params.init_fields.type = 'noise'

    # Forcing parameters
    params.FORCING = True
    params.forcing.type = 'tcrandom_anisotropic'
    params.forcing.forcing_rate = 1.0
    params.forcing.nkmax_forcing = 12
    params.forcing.nkmin_forcing = 10

    # Compute angle
    angle = str(np.round(degrees(np.arcsin(F)), 1))
    params.forcing.tcrandom_anisotropic.angle = angle

    # Compute Lx in order to have kf ~ 1
    params.oper.Lx = params.oper.Ly = Lh = \
        params.forcing.nkmax_forcing * 2 * pi

    # Compute dissipation wave-number
    k_max = ((2 * pi)/Lh) * (params.oper.nx/2)
    coef = 0.9
    if coef > 1:
        raise ValueError('k_diss should be smaller than k_max.')
    k_diss = coef * k_max

    # Hyper-viscosity order q
    q = 8
    nu_q = params.forcing.forcing_rate**(1./3) * (1/k_diss**q)
    if q == 2:
        params.nu_2 = nu_q
    elif q == 4:
        params.nu_4 = nu_q
    elif q == 8:
        params.nu_8 = nu_q
    else:
        raise ValueError('Hyper-viscosity order not implemented.')

    # Compute N
    params.N = R * (params.forcing.forcing_rate**(1./3)/F)

    # Time stepping parameters
    params.time_stepping.USE_CFL = False
    params.time_stepping.USE_T_END = False
    # params.time_stepping.t_end = 100.
    params.time_stepping.it_end = 60
    params.time_stepping.deltat0 = 1e-10

    # Output parameters
    params.output.sub_directory = ''
    params.output.periods_save.phys_fields = 0.1
    params.output.periods_save.spectra = 0.1
    params.output.periods_save.spatial_means = 0.05
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.periods_save.increments = 1.

    params.output.periods_print.print_stdout = 1e-16

    return params


def make_sim(R, F, factor_diss):
    """ Make simulation. """
    params = _make_params_sim(R, F)
    params.nu_8 = params.nu_8 * 10**factor_diss
    params.output.sub_directory = 'bugs'
    sim = Simul(params)
    sim.time_stepping.start()
    fig = sim.output.print_stdout.plot_memory()
    fig.savefig('tmp_fig_memory.png')


if __name__ == '__main__':
    R = 1
    F = 0.5
    factor_diss = 4
    make_sim(R, F, factor_diss)
