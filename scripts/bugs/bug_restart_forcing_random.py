import matplotlib.pyplot as plt

from fluiddyn.util.mpi import printby0

from fluidsim.solvers.ns2d.strat.solver import Simul
from fluidsim import load_for_restart

params = Simul.create_default_params()

params.output.sub_directory = "bugs"

nx = ny = 48
Lx = 3
params.oper.nx = nx
params.oper.ny = ny
params.oper.Lx = Lx
params.oper.Ly = Ly = Lx / nx * ny

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 1.0
params.time_stepping.deltat_max = 0.01

n = 8
C = 1.0
dx = Lx / nx
B = 1
D = 1
eps = 1e-2 * B ** (3 / 2) * D ** (1 / 2)
params.nu_8 = (dx / C) ** ((3 * n - 2) / 3) * eps ** (1 / 3)

printby0(f"nu_8 = {params.nu_8:.3e}")

params.init_fields.type = "noise"
params.init_fields.noise.length = 1.0
params.init_fields.noise.velo_max = 1.0

params.forcing.enable = True
params.forcing.type = "tcrandom"
params.forcing.normalized.constant_rate_of = None
params.forcing.key_forced = "rot_fft"

params.output.periods_print.print_stdout = 1e-1

params.output.periods_save.phys_fields = 0.5
params.output.periods_save.spatial_means = 0.02

sim = Simul(params)
sim.time_stepping.start()


params, Simul = load_for_restart(sim.output.path_run)

params.time_stepping.t_end += 1.0

sim2 = Simul(params)

sim2.time_stepping.start()

sim2.output.spatial_means.plot()

plt.close(1)
plt.show()
