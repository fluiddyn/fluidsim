"""Script for a short simulation with the solver ns3d

with the forcing tcrandom_anisotropic

"""

from fluiddyn.util import mpi

import numpy as np

from fluidsim.solvers.ns3d.solver import Simul

from fluidsim import load_for_restart

params = Simul.create_default_params()

params.output.sub_directory = "bug_ns3d"

# show that these options are not involved
params.no_vz_kz0 = True
params.oper.NO_SHEAR_MODES = True

nx = ny = nz = 64
Lx = 2.0 * np.pi
params.oper.nx = nx
params.oper.ny = ny
params.oper.nz = nz
params.oper.Lx = Lx
params.oper.Ly = Ly = Lx / nx * ny
params.oper.Lz = Lz = Lx / nx * nz
params.oper.coef_dealiasing = 0.66

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 0.5

params.nu_2 = 1e-1

params.init_fields.type = "noise"
params.init_fields.noise.length = 1.0
params.init_fields.noise.velo_max = 0.1

params.forcing.enable = True

params.forcing.type = "tcrandom_anisotropic"
# stronger with
params.forcing.type = "tcrandom_anisotropic"

params.forcing.forcing_rate = 1.0
params.forcing.key_forced = "vx_fft"
params.forcing.nkmin_forcing = 0.9
params.forcing.nkmax_forcing = 4.5

params.forcing.tcrandom.time_correlation = 1.0
params.forcing.tcrandom_anisotropic.angle = np.pi / 4.0
params.forcing.tcrandom_anisotropic.delta_angle = np.pi / 3.0

params.output.periods_print.print_stdout = 1e-10
params.output.periods_save.spatial_means = 1e-10
params.output.periods_save.spectra = 0.01

sim = Simul(params)
sim.time_stepping.start()

# Restart

params, Simul = load_for_restart(sim.output.path_run)
params.time_stepping.t_end += 0.5
sim = Simul(params)
sim.time_stepping.start()

params, Simul = load_for_restart(sim.output.path_run)
params.time_stepping.t_end += 0.1
sim = Simul(params)
sim.time_stepping.start()

mpi.printby0(
    f"""
cd {sim.output.path_run}
ipython --matplotlib -i -c "from fluidsim import load; sim = load(); params = sim.params; sim.output.print_stdout.plot()"

# we can see a discontinuity in time at the largest ky
sim.output.spectra.plot1d_times(tmin=0.45, tmax=0.55, key_k='ky')

# but not in kx and kz!
sim.output.spectra.plot1d_times(tmin=0.45, tmax=0.55, key_k='kz')
sim.output.spectra.plot1d_times(tmin=0.45, tmax=0.55, key_k='kx')
"""
)
