#!/usr/bin/env python
#
# run simul_profile.py
# mpirun -np 8 python simul_profile.py

import pstats
import cProfile

import fluidsim

from fluiddyn.util.mpi import rank


key_solver = 'ns3d'

solver = fluidsim.import_module_solver_from_key(key_solver)
params = solver.Simul.create_default_params()

params.short_name_type_run = 'profile'

n = 512
params.oper.nx = n
params.oper.ny = n
params.oper.nz = n/4
L = 6.
params.oper.Lx = L
params.oper.Ly = L
params.oper.Lz = L

params.oper.coef_dealiasing = 2./3

# params.FORCING = False
# #params.forcing.type_forcing = 'noWAVES'
# params.forcing.nkmax_forcing = 5
# params.forcing.nkmin_forcing = 4
# params.forcing.forcing_rate = 1.


params.nu_8 = 1.

params.time_stepping.deltat0 = 1.e-4
params.time_stepping.USE_CFL = False

params.time_stepping.it_end = 10
params.time_stepping.USE_T_END = False

# params.oper.type_fft = 'fluidfft.fft3d.with_fftw3d'
# params.oper.type_fft = 'fluidfft.fft3d.with_cufft'
params.oper.type_fft = 'fluidfft.fft3d.mpi_with_fftwmpi3d'

# params.init_fields.type_flow_init = 'DIPOLE'


params.output.periods_print.print_stdout = 0

params.output.HAS_TO_SAVE = False
params.output.periods_save.phys_fields = 0.
# params.output.periods_save.spatial_means = 0.
# params.output.periods_save.spectra = 0.
# params.output.periods_save.spect_energy_budg = 0.
# params.output.periods_save.increments = 0.


sim = solver.Simul(params)

# to evaluate the total cpu time...
# from time import clock
# tstart = clock()
# sim.time_stepping.start()
# print('Total cpu time: {} s'.format(clock() - tstart))


# to profile the code...
cProfile.runctx("sim.time_stepping.start()",
                globals(), locals(), "Profile.prof")

if rank == 0:
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats(10)
