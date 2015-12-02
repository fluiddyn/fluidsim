#!/usr/bin/env python
#
# run simul_profile.py
# mpirun -np 8 python simul_profile.py

import pstats
import cProfile

import fluidsim

# key_solver = 'NS2D'
# key_solver = 'SW1l'
# key_solver = 'SW1l.onlywaves'
# key_solver = 'SW1l.exactlin'
key_solver = 'PLATE2D'

solver = fluidsim.import_module_solver_from_key(key_solver)
params = solver.Simul.create_default_params()

params.short_name_type_run = 'profile'

nh = 192
params.oper.nx = nh
params.oper.ny = nh
Lh = 6.
params.oper.Lx = Lh
params.oper.Ly = Lh

params.oper.coef_dealiasing = 2./3

params.FORCING = False
#params.forcing.type_forcing = 'noWAVES'
params.forcing.nkmax_forcing = 5
params.forcing.nkmin_forcing = 4
params.forcing.forcing_rate = 1.


delta_x = Lh/nh
params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8

try:
    params.f = 1.
    params.c2 = 200.
except (KeyError, AttributeError):
    pass

params.time_stepping.deltat0 = 1.e-4
params.time_stepping.USE_CFL = False

params.time_stepping.it_end = 1000
params.time_stepping.USE_T_END = False

#params.oper.type_fft = 'FFTWCY'

# params.init_fields.type_flow_init = 'DIPOLE'


params.output.periods_print.print_stdout = 0

params.output.HAS_TO_SAVE = False
params.output.periods_save.phys_fields = 0.
params.output.periods_save.spatial_means = 0.
params.output.periods_save.spectra = 0.
# params.output.periods_save.spect_energy_budg = 0.
# params.output.periods_save.increments = 0.


sim = solver.Simul(params)
cProfile.runctx("sim.time_stepping.start()",
                globals(), locals(), "Profile.prof")


if sim.oper.rank == 0:
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats(10)
