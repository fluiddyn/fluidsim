from math import pi
import os

from fluidsim.solvers.ns2d.solver import Simul

if os.environ.get("FLUIDSIM_TESTS_EXAMPLES", False):
    sub_directory = "tests_examples"
    nh = 24
else:
    sub_directory = "examples"
    nh = 32

params = Simul.create_default_params()

params.output.sub_directory = sub_directory

params.oper.nx = params.oper.ny = nh
params.oper.Lx = params.oper.Ly = Lh = 2 * pi

delta_x = Lh / nh
params.nu_8 = 2.0 * params.forcing.forcing_rate ** (1.0 / 3) * delta_x**8

params.time_stepping.t_end = 2.0

params.init_fields.type = "dipole"

params.forcing.enable = True
params.forcing.type = "proportional"

params.output.periods_print.print_stdout = 0.25

params.output.periods_save.phys_fields = 1.0
params.output.periods_save.spectra = 0.5
params.output.periods_save.spatial_means = 0.05
params.output.periods_save.spect_energy_budg = 0.5
params.output.periods_save.increments = 0.5

params.output.periods_plot.phys_fields = 0.0

sim = Simul(params)

sim.time_stepping.start()
