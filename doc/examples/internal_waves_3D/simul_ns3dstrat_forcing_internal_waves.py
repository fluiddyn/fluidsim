"""Script for a short simulation with the solver ns3d.strat

with the forcing tcrandom_anisotropic

"""

from fluiddyn.util import mpi

import numpy as np

from fluidsim.solvers.ns3d.strat.solver import Simul

params = Simul.create_default_params()

params.output.sub_directory = "examples"

nx = ny = nz = 64
Lx = 2.0 * np.pi
params.oper.nx = nx
params.oper.ny = ny
params.oper.nz = nz
params.oper.Lx = Lx
params.oper.Ly = Ly = Lx / nx * ny
params.oper.Lz = Lz = Lx / nx * nz

params.time_stepping.USE_T_END = True
params.time_stepping.cfl_coef = 0.2
params.time_stepping.t_end = 50.0

# Brunt Vaisala frequency
params.N = 1.0
params.nu_2 = 1e-1

mpi.printby0(f"nu_2 = {params.nu_2:.3e}")

params.init_fields.type = "noise"
params.init_fields.noise.length = 1.0
params.init_fields.noise.velo_max = 0.01

params.forcing.enable = True
params.forcing.type = "tcrandom_anisotropic"
params.forcing.forcing_rate = 1.0
params.forcing.key_forced = "vp_fft"
params.forcing.tcrandom_anisotropic.angle = np.pi / 4.0
params.forcing.tcrandom_anisotropic.delta_angle = 0.1
params.forcing.tcrandom_anisotropic.kf_min = 1.3
params.forcing.tcrandom_anisotropic.kf_max = 1.5
params.forcing.tcrandom_anisotropic.kz_negative_enable = False
params.forcing.tcrandom.time_correlation = 1.0

params.output.periods_print.print_stdout = 1e-1

params.output.periods_save.phys_fields = 0.5
params.output.periods_save.spatial_means = 0.1
params.output.periods_save.spectra = 0.1
params.output.periods_save.spect_energy_budg = 0.1
params.output.spectra.kzkh_periodicity = 1

sim = Simul(params)
sim.time_stepping.start()


mpi.printby0(
    f"""
# To visualize the output with Paraview, create a file states_phys.xmf with:

fluidsim-create-xml-description {sim.output.path_run}

# To visualize with fluidsim:

cd {sim.output.path_run}
ipython

# in ipython:

from fluidsim import load_sim_for_plot
sim = load_sim_for_plot()
sim.output.phys_fields.set_equation_crosssection('x={Lx/2}')
sim.output.phys_fields.animate('b')

"""
)