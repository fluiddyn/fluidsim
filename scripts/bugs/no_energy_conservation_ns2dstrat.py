"""no_energy_conservation_ns2d_strat.py
=======================================

# Bug description
-----------------

- Values involving the forcing are completely wrong for t = 0

- big difference between d_t E and P - eps when eps is "large" (see comments in
  the code). This may be because we do not compute correctly the forcing if the
  forced scales are dissipative.

- small difference (which tends towards 0 when dt goes to 0 (?)) between d_t E and
  P - eps

Same behaviour with fluiddyn/fluidsim

# To see the bugs
-----------------

python no_energy_conservation_ns2dstrat.py

"""

# import numpy as np
import matplotlib.pyplot as plt

# from fluidsim.solvers.ns2d.strat.solver import Simul
from fluidsim.solvers.ns2d.solver import Simul

params = Simul.create_default_params()

# params.oper.nx = nx = 128*2
# params.oper.ny = ny = nx // 4

params.oper.nx = nx = 64
params.oper.ny = ny = nx

params.oper.Lx = 10.0
params.oper.Ly = params.oper.Lx * (ny / nx)
# params.oper.coef_dealiasing = 0.5

# it is completely wrong!
params.nu_8 = 1e-8
# there is clearly a problem
# params.nu_8 = 1e-10
# nearly ok
# params.nu_8 = 1e-12
# this is ok
# params.nu_8 = 1e-14

# params.N = 3.
params.init_fields.type = "noise"
params.init_fields.noise.length = 1
params.init_fields.noise.velo_max = 1e-4

params.forcing.enable = True
# params.forcing.type = 'tcrandom_anisotropic'
params.forcing.type = "tcrandom"

params.forcing.nkmax_forcing = 10
params.forcing.nkmin_forcing = 4
params.forcing.tcrandom.time_correlation = 1.0
# params.forcing.key_forced = 'ap_fft'
# params.forcing.normalized.which_root = 'minabs'

params.time_stepping.t_end = 2
params.time_stepping.cfl_coef = 0.8
# params.time_stepping.USE_CFL = False
params.time_stepping.deltat0 = 0.02


params.output.HAS_TO_SAVE = True
params.output.periods_save.spatial_means = 1e-10
params.output.periods_save.phys_fields = 0.2
sim = Simul(params)
sim.time_stepping.start()

# sim.output.spatial_means.plot()


try:
    sim.output.spatial_means.plot_dt_enstrophy()
except AttributeError:
    sim.output.spatial_means.plot_dt_energy()

plt.show()
