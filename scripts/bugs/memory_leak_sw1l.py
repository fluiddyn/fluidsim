"""Investigate possible memory leak with sw1l solver

`Related issue <https://foss.heptapod.net/fluiddyn/fluidsim/issues/27>`_

Reproduce
---------
* Requirements: h5py, pythran, transonic

* Run::

    mpirun -np 2 python memory_leak_sw1l.py

"""
from fluidsim.solvers.sw1l.onlywaves.solver import Simul

params = Simul.create_default_params()

nh = 96
params.c2 = 10
params.oper.nx = params.oper.ny = nh
params.oper.type_fft = "fft2d.mpi_with_fftw1d"

params.forcing.enable = True
params.forcing.type = "waves"
params.forcing.forcing_rate = 0.01

params.time_stepping.USE_T_END = False
params.time_stepping.USE_CFL = False
params.time_stepping.it_end = 1_000
params.time_stepping.deltat0 = dt_sim = 0.01

params.init_fields.type = "noise"
params.nu_2 = 0.001 * (96 / nh)

params.output.sub_directory = "test_issue27"
params.output.HAS_TO_SAVE = True

dt_output = dt_sim * 100

params.output.periods_print.print_stdout = dt_output

# prime suspects
# --------------
params.output.periods_save.increments = dt_output / 100
params.output.periods_save.spectra = dt_output / 100

# other outputs
# --------------
#  params.output.periods_save.phys_fields = dt_output
#  params.output.periods_save.spatial_means = dt_output
#  params.output.periods_save.pdf = dt_output
#  params.output.periods_save.spect_energy_budg = dt_output
#  params.output.periods_save.time_signals_fft = dt_output

# only to visualize
# -----------------
# params.output.ONLINE_PLOT_OK = True
# params.output.phys_fields.field_to_plot = "eta"
# params.output.periods_plot.phys_fields = dt_output

sim = Simul(params)
sim.time_stepping.start()
