"""no_energy_conservation_ns2d_strat.py
=======================================

# Bug description
-----------------

- Values involving the forcing are completely wrong for t = 0

- small difference (which tends towards 0 when dt goes to 0 (?)) between d_t E and
  P - eps

Same behaviour with fluiddyn/fluidsim

# To see the bugs
-----------------

python no_energy_conservation_ns2dstrat.py

"""
from __future__ import print_function

# import numpy as np
from math import pi
import matplotlib.pyplot as plt

from fluidsim.solvers.ns2d.strat.solver import Simul
# from fluidsim.solvers.ns2d.solver import Simul

params = Simul.create_default_params()

params.oper.nx = nx = 128

params.oper.ny = ny = nx
params.oper.Lx = 2 * pi
params.oper.Ly = params.oper.Lx * (ny / nx)
# params.oper.coef_dealiasing = 0.66

params.nu_8 = 1e-14
params.N = 1.2
params.init_fields.type = 'noise'
# params.init_fields.noise.velo_max = 1e-10

params.forcing.enable = True
params.forcing.type = 'tcrandom_anisotropic'
# params.forcing.type = 'tcrandom'

params.forcing.nkmax_forcing = 10
params.forcing.nkmin_forcing = 4
params.forcing.tcrandom.time_correlation = 10.
params.forcing.key_forced = 'ap_fft'
# params.forcing.normalized.which_root = 'minabs'

params.time_stepping.t_end = 2
# params.time_stepping.cfl_coef = 0.5
# params.time_stepping.USE_CFL = False
params.time_stepping.deltat0 = 0.02


params.output.HAS_TO_SAVE = True
params.output.periods_save.spatial_means = 1e-10
params.output.periods_save.phys_fields = 0.2
sim = Simul(params)
sim.time_stepping.start()

# sim.output.spatial_means.plot()

sim.output.spatial_means.plot_dt_energy()

try:
    sim.output.spatial_means.plot_dt_enstrophy()
except AttributeError:
    pass

plt.show()
