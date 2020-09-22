"""Script for a short simulation with the solver ns2d.strat

Note how is used the user-defined forcing maker...

"""
from math import pi

import numpy as np

from fluiddyn.util.mpi import rank

from fluidsim.solvers.ns2d.strat.solver import Simul

params = Simul.create_default_params()

params.output.sub_directory = "examples"
params.short_name_type_run = "forcinginscript"

params.oper.nx = nx = 64
params.oper.ny = nx // 2
params.oper.Lx = lx = 2 * pi
params.oper.Ly = lx / 2
params.oper.coef_dealiasing = 0.7


params.nu_8 = 1e-9
params.N = 1.0  # Brunt Vaisala frequency

params.time_stepping.t_end = 10.0

params.init_fields.type = "noise"

params.forcing.enable = True
params.forcing.type = "in_script_coarse"
params.forcing.nkmax_forcing = 12
params.forcing.key_forced = "rot_fft"


params.output.sub_directory = "examples"
params.output.periods_print.print_stdout = 0.5
params.output.periods_save.phys_fields = 0.2
params.output.periods_save.spectra = 0.5
params.output.periods_save.spatial_means = 0.05
params.output.periods_save.spect_energy_budg = 1.0
params.output.periods_save.increments = 1.0

sim = Simul(params)

# monkey-patching for forcing
if rank == 0:
    forcing_maker = sim.forcing.forcing_maker
    oper = forcing_maker.oper_coarse
    forcing0 = 2 * np.cos(2 * pi * oper.Y / oper.ly)
    omega = 2 * pi

    def compute_forcingc_each_time(self):
        return forcing0 * np.sin(omega * sim.time_stepping.t)

    forcing_maker.monkeypatch_compute_forcingc_each_time(
        compute_forcingc_each_time
    )


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

sim.output.phys_fields.animate('b', dt_frame_in_sec=0.1, dt_equations=0.1)
"""
    )
