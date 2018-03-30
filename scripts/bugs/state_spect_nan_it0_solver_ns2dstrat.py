"""
state_spect=nan_it=0_solver_ns2dStrat.py
========================================

# Environment:
-------------
Python 3.6

# Description of the bug:
-------------------------
The state_spect is nan at the initial time of the computation.
To reproduce the bug, it needs to be executed several times.
It has been tested with solver NS2D --> No bug. 
When params.forcing.enable = False --> No bug.
For small spatial resolution, is harder to reproduce the bug. We choose nh=240.

# Traceback:
------------
Traceback (most recent call last):
  File "state_spect=nan_it=0_solver_ns2dStrat.py", line 53, in <module>
    sim.time_stepping.start()
  File "/home/users/calpelin7m/Dev/fluidsim/fluidsim/base/time_stepping/base.py", line 135, in start
    self.one_time_step()
  File "/home/users/calpelin7m/Dev/fluidsim/fluidsim/base/time_stepping/base.py", line 163, in one_time_step
    self.one_time_step_computation()
  File "/home/users/calpelin7m/Dev/fluidsim/fluidsim/solvers/ns2d/strat/time_stepping.py", line 181, in one_time_step_computation
    'nan at it = {0}, t = {1:.4f}'.format(self.it, self.t))
ValueError: nan at it = 0, t = 0.0000

# Steps to reproduce the bug:
-----------------------------
Execute this script several times until you find the bug

python state_spect\=nan_it\=0_solver_ns2dStrat.py

# Save the results:
-------------------
To save the results use
params.output.HAS_TO_SAVE = True

"""

from __future__ import print_function

from fluidsim.solvers.ns2d.strat.solver import Simul
from math import pi, degrees

import numpy as np

params = Simul.create_default_params()
params.oper.nx = params.oper.ny = 240

params.init_fields.type = 'noise'

# Forcing parameters
params.forcing.enable = True
params.forcing.type = 'tcrandom'
params.forcing.nkmax_forcing = 12
params.forcing.nkmin_forcing = 8

# Time stepping parameters
params.time_stepping.USE_CFL = True
params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 0.05

# Output parameters
params.output.HAS_TO_SAVE = False
params.output.periods_print.print_stdout = 0.01

sim = Simul(params)

sim.time_stepping.start()
