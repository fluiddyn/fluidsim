import fluiddyn as fld
from fluidsim.solvers.ns2d.solver import Simul

params = Simul.create_default_params()

params.short_name_type_run = "test"

params.oper.nx = params.oper.ny = nh = 32
params.oper.Lx = params.oper.Ly = Lh = 10

delta_x = Lh / nh
params.nu_8 = 2.0 * params.forcing.forcing_rate ** (1.0 / 3) * delta_x**8

params.time_stepping.t_end = 10.0

params.init_fields.type = "dipole"

params.forcing.enable = True
params.forcing.type = "tcrandom"

params.output.sub_directory = "examples"

params.output.periods_print.print_stdout = 0.5

params.output.periods_save.phys_fields = 1.0
params.output.periods_save.spectra = 0.5
params.output.periods_save.spatial_means = 0.05
params.output.periods_save.spect_energy_budg = 0.5
params.output.periods_save.increments = 0.5

params.output.ONLINE_PLOT_OK = True

params.output.spectra.HAS_TO_PLOT_SAVED = True
params.output.spatial_means.HAS_TO_PLOT_SAVED = True
params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True
params.output.increments.HAS_TO_PLOT_SAVED = True

params.output.phys_fields.field_to_plot = "rot"

sim = Simul(params)

sim.time_stepping.start()
sim.output.phys_fields.plot()

fld.show()
