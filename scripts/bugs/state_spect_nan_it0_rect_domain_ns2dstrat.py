"""
state_spect_nan_it0_rect_domain_ns2dstrat.py
============================================

# Environment:
--------------
python 3.6

# Bug description:
------------------
ValueError: nan at it = 0, t=0.0000 obtained when executing the script.

# Traceback:
------------
Traceback (most recent call last):
  File "state_spect_nan_it0_rect_domain_ns2dstrat.py", line 38, in <module>
    sim.time_stepping.start()
  File "/home/users/calpelin7m/Dev/fluidsim/fluidsim/base/time_stepping/base.py", line 139, in start
    self.one_time_step()
  File "/home/users/calpelin7m/Dev/fluidsim/fluidsim/base/time_stepping/base.py", line 167, in one_time_step
    self.one_time_step_computation()
  File "/home/users/calpelin7m/Dev/fluidsim/fluidsim/solvers/ns2d/strat/time_stepping.py", line 179, in one_time_step_computation
    'nan at it = {0}, t = {1:.4f}'.format(self.it, self.t))
ValueError: nan at it = 0, t = 0.0000

# Notes:
--------
- Bug in the cluster in sequential
- Bug in the cluster in MPI
- Bug in PC in sequential
- Bug in PC in MPI

- When params.forcing.enable = False --> No bug

# To run the bug:
-----------------
python state_spect_nan_it0_rect_domain_ns2dstrat.py
"""
from __future__ import print_function

from fluidsim.solvers.ns2d.strat.solver import Simul
from math import pi, degrees

import numpy as np

params = Simul.create_default_params()
params.oper.nx = 128
params.oper.ny = params.oper.nx // 4

params.oper.Lx = 2 * pi
params.oper.Ly = params.oper.Lx * (params.oper.ny / params.oper.nx)

params.init_fields.type = 'noise'

# Forcing parameters
params.forcing.enable = True
params.forcing.type = 'tcrandom_anisotropic'
params.forcing.nkmax_forcing = 12
params.forcing.nkmin_forcing = 8

# Time stepping parameters
params.time_stepping.USE_CFL = True
params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 0.05

# Output parameters
params.output.HAS_TO_SAVE = False
params.output.periods_print.print_stdout = 1e-15

sim = Simul(params)

sim.time_stepping.start()
