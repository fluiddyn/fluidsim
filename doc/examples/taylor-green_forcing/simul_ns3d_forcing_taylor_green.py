from fluidsim.solvers.ns3d.solver import Simul

import matplotlib.pyplot as plt

params = Simul.create_default_params()

params.output.sub_directory = "examples"

N = 64
F0 = 1.0
V0 = 1.0
L = 1.0
T = 10.0

params.oper.nx = params.oper.ny = params.oper.nz = N
params.oper.Lx = params.oper.Ly = params.oper.Lz = L
params.nu_2 = 0.01

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = T * L / V0

params.init_fields.type = "noise"
params.init_fields.noise.velo_max = 0.001

params.forcing.enable = True
params.forcing.type = "taylor_green"
params.forcing.key_forced = None

taylor_green = params.forcing.taylor_green
taylor_green.amplitude = F0

params.output.periods_print.print_stdout = 1e-1

params.output.periods_print.print_stdout = 1.0
params.output.periods_save.phys_fields = 0.5
params.output.periods_save.spectra = 0.5
params.output.periods_save.spatial_means = 0.1

sim = Simul(params)
sim.time_stepping.start()

sim.output.spatial_means.plot()
plt.show()

sim.output.phys_fields.plot("vx")
plt.show()
