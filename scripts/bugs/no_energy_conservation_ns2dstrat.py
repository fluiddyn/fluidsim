"""
no_energy_conservation_ns2d_strat.py
=======================================

# Bug description
-----------------
No energy conservation (no viscosity and no forcing)
params.forcing.enable = False
params.nu_8 = 0

Injection of energy fluctuates.
params.forcing.enable = True

# Notes
--------
Same problem with fluiddyn/fluidsim

# To run the bug
--------------
python no_energy_conservation_ns2dstrat.py

"""
from __future__ import print_function

import numpy as np
from math import pi
from fluidsim.solvers.ns2d.strat.solver import Simul

params = Simul.create_default_params()

params.oper.nx = nx = 128
params.oper.ny = ny = nx 
params.oper.Lx = 2 * pi
params.oper.Ly = params.oper.Lx * (ny / nx)

params.init_fields.type = 'noise'

params.forcing.enable = False
params.forcing.type = 'tcrandom_anisotropic'
params.forcing.nkmax_forcing = 12
params.forcing.nkmin_forcing = 8

params.time_stepping.t_end = 10

params.output.HAS_TO_SAVE = True
params.output.periods_save.spatial_means = 1e-3
sim = Simul(params)
sim.time_stepping.start()

sim.output.spatial_means.plot()


