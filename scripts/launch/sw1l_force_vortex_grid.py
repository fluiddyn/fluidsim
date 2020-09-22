#!/usr/bin/env python
# mpirun -np 4 python simul_force_vortex_grid.py

from math import pi
from fluidsim.solvers.sw1l.solver import Simul
from fluidsim.util.util import mpi

params = Simul.create_default_params()
params.short_name_type_run = "vortex_grid"
params.output.sub_directory = "beskow_tests"

# --------Grid parameters---------
params.oper.nx = params.oper.ny = nh = 512
params.oper.Lx = params.oper.Ly = Lh = 2 * pi
delta_x = Lh / nh

# -----Numerical method parameters----
params.f = 0.0
params.c2 = 100
params.oper.coef_dealiasing = 2.0 / 3
params.time_stepping.t_end = 100.0
params.init_fields.type = "vortex_grid"

# ----- Forcing parameters -----------
params.FORCING = True
params.forcing.type = "waves"
params.forcing.nkmax_forcing = 8
params.forcing.nkmin_forcing = 5
params.forcing.forcing_rate = 1.0

# k_max = pi / delta_x * params.oper.coef_dealiasing # Smallest resolved scale
# k_d = k_max / 2.5
# length_scale = pi / k_d
# params.nu_8 = params.forcing.forcing_rate**(1./3) * length_scale ** (22./3)

# ------Save file / StdOut parameters------------
params.output.periods_print.print_stdout = 0.1
params.output.period_refresh_plots = 0.5
params.output.periods_plot.phys_fields = 0.0
# params.output.spatial_means.HAS_TO_PLOT_SAVED = True
# params.output.spectra.HAS_TO_PLOT_SAVED = True
# params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True

params.output.periods_save.phys_fields = 1.0
params.output.periods_save.spectra = 0.1
params.output.periods_save.spatial_means = 0.1
params.output.periods_save.spect_energy_budg = 0.5
params.output.periods_save.increments = 0.5

# -----Preprocess parameters------------
params.preprocess.enable = True
params.preprocess.forcing_const = 5e4
# params.preprocess.forcing_scale = 'unity'
params.preprocess.forcing_scale = "enstrophy"
params.preprocess.viscosity_const = 1.5
params.preprocess.viscosity_scale = "enstrophy_forcing"
params.preprocess.viscosity_type = "hyper8"

sim = Simul(params)
sim.output.print_stdout(
    "Froude number ~ {:3e}, Viscosity = {:3e} + {:3e}".format(
        (1.0 / params.c2), sim.params.nu_2, sim.params.nu_8
    )
)
sim.output.print_stdout(f"Forcing rate = {sim.params.forcing.forcing_rate}")

sim.time_stepping.start()
