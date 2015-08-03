#!/usr/bin/env python
#coding=utf8
#

import ipdb
import numpy as np
import fluiddyn as fld
from fluiddyn.util import mpi

# key_solver = 'SW1l'
key_solver = 'SW1l.onlywaves'
# key_solver = 'SW1l.exactlin'
solver = fld.simul.import_module_solver_from_key(key_solver)
params = solver.Simul.create_default_params()

params.short_name_type_run = 'pierre_f_nu8_2e2_dealias_89'
params.init_fields.type = 'constant'
params.init_fields.constant.value = 0.

# --------Grid parameters---------
nh = 256
Lh = 2.*np.pi
params.oper.nx = nh
params.oper.ny = nh
params.oper.Lx = Lh
params.oper.Ly = Lh
delta_x = Lh/nh

# -----Numerical method parameters----
params.f = 0.
params.c2 = 10.**2
params.oper.coef_dealiasing = 8./9
params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 20.
params.time_stepping.it_end = 2000
params.time_stepping.USE_CFL = False
params.time_stepping.deltat0 = 1e-3

# ----- Forcing parameters -----------
params.FORCING = True
if params.FORCING:
    params.forcing.type = 'waves'
    #    params.forcing.key_forced = 'eta_fft'
    params.forcing.nkmax_forcing = 8
    params.forcing.nkmin_forcing = 5
    params.forcing.forcing_rate  = 1.
    k_max = np.pi / delta_x * params.oper.coef_dealiasing # Smallest resolved scale
    k_d = k_max / 50.
    params.nu_8 = 2.e2*params.forcing.forcing_rate**(1./3)*delta_x**8
    #params.nu_2 = params.forcing.forcing_rate ** (1./3) / k_d ** (4./3)

# ------Save file / StdOut parameters------------
params.output.periods_save.spatial_means = 0.0
params.output.periods_save.phys_fields = 1.
params.output.periods_save.spectra = 0.1
params.output.periods_save.spect_energy_budg = 0.1
params.output.periods_save.increments = 1.

params.output.periods_print.print_stdout = 0.1

# -------Plotting parameters---------------
params.output.ONLINE_PLOT_OK = False
params.output.spatial_means.HAS_TO_PLOT_SAVED = True
params.output.spectra.HAS_TO_PLOT_SAVED = False

params.output.periods_plot.phys_fields = 1.
params.output.phys_fields.field_to_plot = 'eta'

# -------Simulation commences-------------
sim = solver.Simul(params)
# params = sim.set_viscosity_scale(C=11, scale='FORCING')

if mpi.rank == 0:
    print "Froude number ~ {0:3e}, Viscosity = {1:3e} + {2:3e}".format(
          (1./params.c2), sim.params.nu_2, sim.params.nu_8)
sim.time_stepping.start()

# fld.show()
