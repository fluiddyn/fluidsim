"""Script for a short simulation with the solver ns2d.bouss

The field initialization is done in the script.

"""
from math import pi

import numpy as np

from fluiddyn.util.mpi import rank

from fluidsim.solvers.ns2d.bouss.solver import Simul

params = Simul.create_default_params()

params.oper.nx = nx = 64
params.oper.ny = nx // 2
params.oper.Lx = lx = 2
params.oper.Ly = lx
params.oper.coef_dealiasing = 0.7

params.nu_8 = 1e-10

params.time_stepping.t_end = 5.0

params.init_fields.type = "in_script"

params.output.sub_directory = "examples"
params.output.periods_print.print_stdout = 0.5
params.output.periods_save.phys_fields = 0.1
params.output.periods_save.spatial_means = 0.1

sim = Simul(params)

# field initialization in the script
rot = 1e-6 * sim.oper.create_arrayX_random()
X = sim.oper.X
Y = sim.oper.Y
x0 = y0 = 1.0
R2 = (X - x0) ** 2 + (Y - y0) ** 2
r0 = 0.2
b = -np.exp(-R2 / r0**2)
sim.state.init_from_rotb(rot, b)

# In this case (params.init_fields.type = 'in_script') if we want to plot the
# result of the initialization before the time_stepping, we need to manually
# initialized the output:
#
# sim.output.init_with_initialized_state()
# sim.output.phys_fields.plot(field='b')

sim.time_stepping.start()

if rank == 0:
    print(
        "\nTo display a video of this simulation, you can do:\n"
        f"cd {sim.output.path_run}"
        + """
ipython

# then in ipython (copy the 3 lines in the terminal):

from fluidsim import load_sim_for_plot
sim = load_sim_for_plot()

sim.output.phys_fields.animate('b', dt_frame_in_sec=0.3, dt_equations=0.1)
"""
    )
