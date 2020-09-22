"""
init_linear_mode.py
====================

26/07/2018
Launch simulation initialized by the linear mode.

Check direction of propagation of linear mode ap_fft and
am_fft.

Switch the linear mode with the parameter:
params.init_fields.linear_mode.eigenmode = "ap_fft"

"""

import numpy as np
import matplotlib.pyplot as plt

from math import pi

from fluidsim.solvers.ns2d.strat.solver import Simul

params = Simul.create_default_params()

params.N = 50.0
params.oper.nx = nx = 128
params.oper.ny = ny = nx // 4

# Parameters time stepping
params.time_stepping.USE_CFL = True
params.time_stepping.t_end = 2.0

# Field initialization in the script
params.init_fields.type = "linear_mode"
params.init_fields.linear_mode.eigenmode = "ap_fft"
params.init_fields.linear_mode.i_mode = (2, 1)
params.init_fields.linear_mode.delta_k_adim = 1

# Parameters output
params.output.sub_directory = "tests"
params.output.periods_save.phys_fields = 2e-2

sim = Simul(params)

# # To plot figure before launching simulation.
# sim.output.init_with_initialized_state()
# sim.output.phys_fields.plot("ux")

sim.time_stepping.start()
sim.output.phys_fields.animate()
