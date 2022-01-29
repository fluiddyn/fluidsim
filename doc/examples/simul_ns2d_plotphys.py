import fluiddyn as fld
from fluidsim.solvers.ns2d.solver import Simul

params = Simul.create_default_params()

params.short_name_type_run = "test"

params.oper.nx = params.oper.ny = nh = 32 * 2
params.oper.Lx = params.oper.Ly = Lh = 10.0

delta_x = Lh / nh
params.nu_8 = 2e-3 * params.forcing.forcing_rate ** (1.0 / 3) * delta_x**8

params.time_stepping.t_end = 10.0

params.init_fields.type = "dipole"

params.forcing.enable = True
params.forcing.type = "tcrandom"

params.output.sub_directory = "examples"

params.output.periods_plot.phys_fields = 0.1
params.output.periods_save.phys_fields = 0.2
params.output.periods_save.spatial_means = 0.05

params.output.ONLINE_PLOT_OK = True


sim = Simul(params)
sim.time_stepping.start()

print(
    """
A movie can be produced with the command (using ffmpeg):

sim.output.phys_fields.animate(dt_frame_in_sec=0.1, dt_equations=0.08, repeat=False, save_file=1, tmax=10)
"""
)

fld.show()
