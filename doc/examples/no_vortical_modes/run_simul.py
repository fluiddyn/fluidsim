"""Script for a short simulation with the solver ns3d.strat

with the forcing tcrandom_anisotropic and NO_VORTICAL_MODES = True

"""

from fluiddyn.util import mpi

import numpy as np

from fluidsim.solvers.ns3d.strat.solver import Simul

params = Simul.create_default_params()

kind = "vert"

keys_versus_kind = {
    "toro": "vt_fft",
    "polo": "vp_fft",
    "buoyancy": "b_fft",
    "vert": "vz_fft",
}

params.output.sub_directory = "examples/no_vortical_modes"
params.short_name_type_run = "aniso_" + kind

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

params.NO_VORTICAL_MODES = True
params.N = 1.0
params.nu_2 = 1e-1

mpi.printby0(f"nu_2 = {params.nu_2:.3e}")

params.init_fields.type = "noise"
params.init_fields.noise.length = 1.0
params.init_fields.noise.velo_max = 0.01

params.forcing.enable = True
params.forcing.type = "tcrandom_anisotropic"
params.forcing.forcing_rate = 1.0
params.forcing.key_forced = keys_versus_kind[kind]
params.forcing.nkmin_forcing = 2.9
params.forcing.nkmax_forcing = 7.2

params.forcing.tcrandom.time_correlation = 1.0
params.forcing.tcrandom_anisotropic.angle = np.pi / 3.0
params.forcing.tcrandom_anisotropic.delta_angle = np.pi / 6.0
params.forcing.tcrandom_anisotropic.kz_negative_enable = True

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
ipython --matplotlib -i -c "from fluidsim import load; sim = load()"

# in IPython:

sim.output.phys_fields.set_equation_crosssection('x={Lx/2}')
sim.output.phys_fields.animate('b')
"""
)
