# !/usr/bin/env python
# coding=utf8
#
# run simul_test_conserveE.py
# mpirun -np 8 python simul_test_conserveE.py

key_solver = "NS2D"
# key_solver = 'SW1l'
# key_solver = 'SW1l.onlywaves'
# key_solver = 'SW1l.exactlin'
# key_solver = 'SW1l.modified'

import numpy as np
import fluidsim

solver = fluidsim.import_module_solver_from_key(key_solver)
params = solver.Simul.create_default_params()

params.short_name_type_run = "conservE"

nh = 16 * 4 * 8
Lh = 2 * np.pi
params.oper.nx = nh
params.oper.ny = nh
params.oper.Lx = Lh
params.oper.Ly = Lh

params.oper.coef_dealiasing = 2.0 / 3

params.nu_8 = 0.0
params.nu_4 = 0.0
params.nu_2 = 0.0
params.nu_m4 = 0.0

try:
    params.f = 1.0
    params.c2 = 200.0
except KeyError:
    pass

params.time_stepping.USE_CFL = False

params.time_stepping.it_end = 5
params.time_stepping.USE_T_END = False

params.time_stepping.type_time_scheme = "RK4"

params.init_fields.type_flow_init = "DIPOLE"


params.output.periods_print.print_stdout = 10.0e-15

params.output.periods_save.phys_fields = 0.0
params.output.periods_save.spectra = 0.0
params.output.periods_save.spect_energy_budg = 0.0
params.output.periods_save.increments = 0.0

try:
    params.output.periods_save.pdf = 0.0
    params.output.periods_save.time_signals_fft = False
except KeyError:
    pass


params.output.periods_plot.phys_fields = 0.0


params.time_stepping.deltat0 = 1.0e-1
sim = solver.Simul(params)
sim.time_stepping.start()

params.time_stepping.deltat0 = 1.0e-2
sim = solver.Simul(params)
sim.time_stepping.start()
