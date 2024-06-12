import argparse
import os

from fluiddyn.util.mpi import printby0

from fluidsim.solvers.ns3d.solver import Simul

parser = argparse.ArgumentParser()

parser.add_argument(
    "--nx",
    type=int,
    default=96,
    help="Number of grid points in the x direction.",
)
parser.add_argument(
    "--t_end", type=float, default=8.0, help="End time of the simulation"
)
parser.add_argument(
    "--order",
    type=int,
    default=4,
    help="Order of the viscosity (`2` corresponds to standard viscosity)",
)

args = parser.parse_args()

if "FLUIDSIM_TESTS_EXAMPLES" in os.environ:
    t_end = 1.0
    nx = 24
else:
    t_end = args.t_end
    nx = args.nx

params = Simul.create_default_params()

params.output.sub_directory = "examples"

ny = nz = nx
Lx = 3
params.oper.nx = nx
params.oper.ny = ny
params.oper.nz = nz
params.oper.Lx = Lx
params.oper.Ly = Ly = Lx / nx * ny
params.oper.Lz = Lz = Lx / nx * nz

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = t_end

order_visco = args.order
dx = Lx / nx
epsilon = 1.0
C = 1.0
nu = (dx / C) ** ((3 * order_visco - 2) / 3) * epsilon ** (1 / 3)
setattr(params, f"nu_{order_visco}", nu)

printby0(f"nu_{order_visco} = {nu:.3e}")

params.init_fields.type = "noise"
params.init_fields.noise.length = 1.0
params.init_fields.noise.velo_max = 0.1

params.forcing.enable = True
params.forcing.type = "tcrandom"
params.forcing.normalized.constant_rate_of = None
params.forcing.nkmin_forcing = 3
params.forcing.nkmax_forcing = 4
# solenoidal field (toroidal + poloidal)
params.forcing.key_forced = ["vt_fft", "vp_fft"]
# forcing rate **per key forced**
params.forcing.forcing_rate = 0.5 * epsilon

params.output.periods_print.print_stdout = 1e-1

params.output.periods_save.phys_fields = 0.5
params.output.periods_save.spatial_means = 0.1
params.output.periods_save.spectra = 0.1
params.output.periods_save.spect_energy_budg = 0.1

params.output.spectra.kzkh_periodicity = 1

sim = Simul(params)
sim.time_stepping.start()
