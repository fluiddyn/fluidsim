"""Script for a short simulation with the solver ns2d.strat

"""
import numpy as np

from fluiddyn.util import mpi

from fluidsim.solvers.ns2d.strat.solver import Simul

params = Simul.create_default_params()

params.output.sub_directory = "examples"
params.short_name_type_run = "forcing_const_energy_rate"

coef_box = 4

params.oper.nx = nx = 128
params.oper.ny = nx // coef_box
params.oper.Lx = lx = 2 * np.pi
params.oper.Ly = lx / coef_box
params.oper.coef_dealiasing = 0.65

params.nu_8 = 1e-9
params.N = 4.0  # Brunt Vaisala frequency

params.time_stepping.t_end = 10.0
params.time_stepping.cfl_coef = 0.2
params.time_stepping.max_elapsed = "00:00:05"

params.init_fields.type = "noise"
params.init_fields.noise.velo_max = 0.01

# forcing params
key_forced = "rot_fft"
# key_forced = "ap_fft"
neg_enable = True
# neg_enable = False

params.forcing.enable = True
params.forcing.nkmax_forcing = 8
params.forcing.type = "tcrandom"
forcing_rate = 1.25
params.forcing.forcing_rate = forcing_rate
params.forcing.key_forced = key_forced
params.forcing.tcrandom_anisotropic.kz_negative_enable = neg_enable
params.forcing.normalized.constant_rate_of = "energy"

params.output.sub_directory = "examples"
params.output.periods_print.print_stdout = 0.5
params.output.periods_save.phys_fields = 0.2
params.output.periods_save.spectra = 0.5
params.output.periods_save.spatial_means = 0.05
params.output.periods_save.spect_energy_budg = 1.0
params.output.periods_save.increments = 1.0

sim = Simul(params)

sim.time_stepping.start()

if mpi.rank == 0:
    # Does the energy injection rate have the correct value at all times ?
    means = sim.output.spatial_means.load()
    P_tot = means["PK_tot"] + means["PA_tot"]
    assert np.allclose(P_tot, forcing_rate)
    print(
        f"P_tot corrects when forcing in {key_forced} (neg_enable = {neg_enable})"
    )

mpi.printby0(
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
