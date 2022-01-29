#!/usr/bin/env python
# coding=utf8
#

from math import pi
from fluidsim.solvers.sw1l.solver import Simul
from fluidsim.util.util import mpi

params = Simul.create_default_params()

params.short_name_type_run = "no_flow"
params.init_fields.type = "constant"
params.init_fields.constant[...] = 0.0
params.output.sub_directory = "spect_energy_budg_tests_noflow"

# --------Grid parameters---------
nh = 256
Lh = 2.0 * pi
params.oper.nx = nh
params.oper.ny = nh
params.oper.Lx = Lh
params.oper.Ly = Lh
delta_x = Lh / nh

# -----Numerical method parameters----
params.f = 0.0
params.c2 = 10.0**2
params.oper.coef_dealiasing = 8.0 / 9
params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 5.0
params.time_stepping.it_end = 2000
params.time_stepping.USE_CFL = True
params.time_stepping.deltat0 = 1e-3

# ----- Forcing parameters -----------
params.FORCING = True
if params.FORCING:
    params.forcing.type = "waves"
    params.forcing.nkmax_forcing = 8
    params.forcing.nkmin_forcing = 5
    params.forcing.forcing_rate = 1.0
    k_max = pi / delta_x * params.oper.coef_dealiasing  # Smallest resolved scale
    k_d = k_max / 2.5
    length_scale = pi / k_d
    params.nu_8 = params.forcing.forcing_rate ** (1.0 / 3) * length_scale ** (
        22.0 / 3
    )
    # params.nu_2 = params.forcing.forcing_rate ** (1./3) * length_scale ** (4./3)

# ------Save file / StdOut parameters------------
params.output.periods_save.spatial_means = 0.1
params.output.periods_save.phys_fields = 1.0
params.output.periods_save.spectra = 0.1
params.output.periods_save.spect_energy_budg = 0.1
params.output.periods_save.increments = 0.5

params.output.periods_print.print_stdout = 0.1

# -------Plotting parameters---------------
# params.output.ONLINE_PLOT_OK = True
# params.output.spatial_means.HAS_TO_PLOT_SAVED = True
# params.output.spectra.HAS_TO_PLOT_SAVED = True

# params.output.periods_plot.phys_fields = 1.
# params.output.phys_fields.field_to_plot = 'eta'

# -------Preprocess parameters---------------
params.preprocess.enable = False
params.preprocess.init_field_scale = "unity"
params.preprocess.forcing_const = 1.0
params.preprocess.forcing_scale = "unity"
params.preprocess.viscosity_const = 2.5
params.preprocess.viscosity_scale = "forcing"
params.preprocess.viscosity_type = "hyper8"

# -------Simulation commences-------------
sim = Simul(params)
if mpi.rank == 0:
    print(
        "Froude number ~ {:3e}, Viscosity = {:3e} + {:3e}".format(
            (1.0 / params.c2), sim.params.nu_2, sim.params.nu_8
        )
    )

sim.time_stepping.start()
