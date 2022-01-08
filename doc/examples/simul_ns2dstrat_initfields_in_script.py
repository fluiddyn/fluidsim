"""Script for a short simulation with the solver ns2d.strat

The field initialization is done in the script.

"""
import numpy as np

from fluiddyn.util.mpi import rank

from fluidsim.solvers.ns2d.strat.solver import Simul

params = Simul.create_default_params()

params.oper.nx = nx = 64
params.oper.ny = nx // 2
params.oper.Ly = ly = 2
params.oper.Lx = lx = 4 * ly
params.oper.coef_dealiasing = 0.7

params.nu_8 = 0.0

params.time_stepping.t_end = 10.0

params.init_fields.type = "in_script"

params.output.sub_directory = "examples"
params.output.periods_print.print_stdout = 0.5
params.output.periods_save.phys_fields = 0.1
params.output.periods_save.spatial_means = 0.1

sim = Simul(params)

# field initialization in the script
dx = lx / nx
width = max(4 * dx, 0.02)


def step_func(x):
    """Activation function"""
    return 0.5 * (np.tanh(x / width) + 1)


Y = sim.oper.Y
ux = (
    5e-2 * sim.oper.create_arrayX_random()
    + step_func((Y - ly / 4) / ly)
    - step_func((Y - 3 * ly / 4) / ly)
    - 1 / 2
)

uy = 5e-2 * sim.oper.create_arrayX_random()

sim.state.init_statespect_from(ux_fft=sim.oper.fft(ux), uy_fft=sim.oper.fft(uy))

# In this case (params.init_fields.type = 'in_script') if we want to plot the
# result of the initialization before the time_stepping, we need to manually
# initialized the output:
#
# sim.output.init_with_initialized_state()
# sim.output.phys_fields.plot(field="ux")

# import sys

# sys.exit()

sim.time_stepping.start()


if rank == 0:

    print(
        "\nTo display a video of this simulation, you can do:\n"
        f"cd {sim.output.path_run}"
        + """
ipython --matplotlib -i -c "from fluidsim import load; sim = load()"

# then in ipython (copy the line in the terminal):

sim.output.phys_fields.animate('b', dt_frame_in_sec=0.3, dt_equations=0.1)
"""
    )

    sim.output.phys_fields.plot(field="uy")

    import matplotlib.pyplot as plt

    plt.show()
