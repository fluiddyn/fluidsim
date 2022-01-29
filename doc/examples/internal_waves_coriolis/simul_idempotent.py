"""Simulation similar to experiments in Coriolis with two plates oscillating to
force internal waves in the Coriolis platform.

Launch with::

  mpirun -np 4 python simul_idempotent.py 0.02 30

or::

  mpirun -np 4 python simul_idempotent.py 0.02 30 --max-elapsed 00:02:00

"""

from math import pi
from pathlib import Path
import sys
import argparse

from fluiddyn.util import mpi
from fluidsim import FLUIDSIM_PATH, load_for_restart
from fluidsim.solvers.ns3d.strat.solver import Simul

parser = argparse.ArgumentParser()
parser.add_argument("amplitude", help="Amplitude of the movement", type=float)
parser.add_argument("nz", help="Number of point over the z direction", type=int)
parser.add_argument(
    "-me",
    "--max-elapsed",
    help="Maximum elapsed time",
    type=str,
    default="00:02:00",
)

args = parser.parse_args()
mpi.printby0(args)

amplitude = args.amplitude  # m
nz = args.nz
max_elapsed = args.max_elapsed

# main input parameters
omega_f = 0.3  # rad/s
delta_omega_f = 0.03  # rad/s
N = 0.4  # rad/s

aspect_ratio = 4
nx = ny = nz * aspect_ratio

sub_directory = "waves_coriolis"
short_name_type_run = f"ampl{amplitude}"

path_dir = Path(FLUIDSIM_PATH) / sub_directory

path_idempotent_file = (
    path_dir / f"idempotent_{short_name_type_run}_{nx}x{ny}x{nz}.txt"
)

path_idempotent_file_exists = None
if mpi.rank == 0:
    path_idempotent_file_exists = path_idempotent_file.exists()
if mpi.nb_proc > 1:
    path_idempotent_file_exists = mpi.comm.bcast(
        path_idempotent_file_exists, root=0
    )

if not path_idempotent_file_exists:
    mpi.printby0("New simulation")
    # useful parameters and secondary input parameters
    period_N = 2 * pi / N
    # total period of the forcing signal
    period_forcing = 1e3 * period_N

    params = Simul.create_default_params()

    params.short_name_type_run = short_name_type_run

    params.N = N

    params.output.sub_directory = sub_directory

    lz = 1

    params.oper.nx = nx
    params.oper.ny = ny
    params.oper.nz = nz
    params.oper.Lx = lx = lz / nz * nx
    params.oper.Ly = ly = lz / nz * ny
    params.oper.Lz = lz
    params.oper.NO_SHEAR_MODES = True
    params.no_vz_kz0 = True

    r"""

    Order of magnitude of nu_8?
    ---------------------------

    Since the dissipation frequency is $\nu_n k^n$, we can define a Reynolds number
    as:

    $$Re_n = \frac{U L^{n-1}}{\nu_n}.$$

    If we take a turbulent scaling $u(l) = (\varepsilon l)^{1/3}$, we obtain

    $$Re_n(l) = \frac{\varepsilon^{1/3} l^(n - 2/3)}{\nu_n}.$$

    The Kolmogorov length scale $\eta_n$ can be defined as the scale for which
    $Re_n(l) = 1$:

    $$ {\eta_n}^{n - 2/3} = \frac{\varepsilon^{1/3}}{\nu_n} $$

    We want that $dx < \eta_n$, so we choose $\nu_n$ such that $dx = C \eta_n$
    where $C$ is a constant of order 1.

    """
    n = 2
    C = 1.0
    dx = lx / nx
    U = amplitude * omega_f
    H = 1
    eps = 1e-2 * U**3 / H
    params.nu_2 = (dx / C) ** ((3 * n - 2) / 3) * eps ** (1 / 3)

    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = 20 * period_N
    params.time_stepping.deltat_max = deltat_max = period_N / 40
    params.time_stepping.max_elapsed = max_elapsed

    params.init_fields.type = "noise"
    params.init_fields.noise.velo_max = 0.001
    params.init_fields.noise.length = 2e-1

    params.forcing.enable = True
    params.forcing.type = "watu_coriolis"

    watu = params.forcing.watu_coriolis
    watu.omega_f = omega_f
    watu.delta_omega_f = delta_omega_f
    watu.amplitude = amplitude
    watu.period_forcing = period_forcing
    watu.approximate_dt = period_N / 1e1
    watu.nb_wave_makers = 2

    params.output.periods_print.print_stdout = 4.0

    params.output.periods_save.phys_fields = 16.0
    params.output.periods_save.spectra = 4.0
    params.output.periods_save.spatial_means = 4.0
    params.output.periods_save.spect_energy_budg = 4.0

    params.output.spectra.kzkh_periodicity = 2

    sim = Simul(params)
    if mpi.rank == 0:
        print(f"Creating file {path_idempotent_file}")
        with open(path_idempotent_file, "w") as file:
            file.write(sim.name_run)

else:
    mpi.printby0(
        f"file {path_idempotent_file} exists:\nrestarting the simulation"
    )

    name_dir = None
    if mpi.rank == 0:
        with open(path_idempotent_file) as file:
            name_dir = file.read().strip()
        print(str(path_dir / name_dir))
    if mpi.nb_proc > 1:
        name_dir = mpi.comm.bcast(name_dir, root=0)

    params, Simul = load_for_restart(path_dir / name_dir)
    sim = Simul(params)

sim.time_stepping.start()

if sim.time_stepping._has_to_stop:
    mpi.printby0("Simulation is not completed. The script should be relaunched")
    sys.exit(99)

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
sim.output.phys_fields.set_equation_crosssection('x={sim.params.oper.Lx/2}')
sim.output.phys_fields.animate('b')

"""
)
