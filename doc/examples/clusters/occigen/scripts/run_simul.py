"""
# A simple MILESTONE simulation

python run_simul.py -D 0.5 -s 0.1 -N 0.5 -nypc 60 --n_periods 5 --only_init

"""

from math import pi
import argparse

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.strat.solver import Simul
from fluidsim.base.forcing.milestone import PeriodicUniform


parser = argparse.ArgumentParser()
parser.add_argument("-N", type=float, default=0.5, help="Brunt–Väisälä frequency")

parser.add_argument(
    "-D", "--diameter", type=float, default=0.5, help="Diameter of the cylinders"
)

parser.add_argument(
    "-s",
    "--speed",
    type=float,
    default=0.1,
    help="Maximum speed of the cylinders",
)

parser.add_argument(
    "-nc", "--number-cylinders", type=int, default=3, help="Number of cylinders"
)

parser.add_argument(
    "-nypc",
    "--ny-per-cylinder",
    type=int,
    default=60,
    help="Number of numerical points for one cylinder",
)

parser.add_argument(
    "-np", "--n-periods", type=int, default=4, help="Number of periods"
)

parser.add_argument(
    "-oi",
    "--only-init",
    action="store_true",
    help="Only run initialization phase",
)

parser.add_argument(
    "-opp",
    "--only-print-params",
    action="store_true",
    help="Only print parameters",
)

parser.add_argument(
    "-me",
    "--max-elapsed",
    help="Maximum elapsed time",
    type=str,
    default="00:02:00",
)


def main(args):

    params = Simul.create_default_params()

    diameter = args.diameter  # m
    speed = args.speed  # m/s
    params.N = args.N  # rad/s

    mesh = 3 * diameter
    number_cylinders = args.number_cylinders

    ly = params.oper.Ly = mesh * number_cylinders
    lx = params.oper.Lx = 1.5 * 5
    ny = params.oper.ny = args.ny_per_cylinder * number_cylinders

    nx_float = ny * lx / ly
    nx = params.oper.nx = round(nx_float)
    assert nx == nx_float

    dx = lx / nx

    lz = params.oper.Lz = mesh
    params.oper.nz = round(lz / dx)

    params.oper.NO_SHEAR_MODES = True

    params.forcing.enable = True
    params.forcing.type = "milestone"
    params.forcing.milestone.nx_max = 64
    objects = params.forcing.milestone.objects

    objects.number = number_cylinders
    objects.diameter = diameter
    objects.width_boundary_layers = 0.1

    movement = params.forcing.milestone.movement

    movement.type = "periodic_uniform"
    movement.periodic_uniform.length = lx - 1.0
    movement.periodic_uniform.length_acc = 0.25
    movement.periodic_uniform.speed = speed

    params.init_fields.type = "noise"
    params.init_fields.noise.velo_max = speed / 2
    params.init_fields.noise.length = diameter

    movement = PeriodicUniform(
        speed,
        movement.periodic_uniform.length,
        movement.periodic_uniform.length_acc,
        lx,
    )

    params.time_stepping.t_end = movement.period * args.n_periods
    params.time_stepping.deltat_max = 0.1 * diameter / speed
    params.time_stepping.max_elapsed = args.max_elapsed

    params.nu_2 = 1e-6

    epsilon_eval = 0.02 * speed**3 / mesh
    eta_elav = (params.nu_2**3 / epsilon_eval) ** (1 / 4)

    kmax = params.oper.coef_dealiasing * pi / dx
    eta_kmax = 2 * pi / kmax
    nu_2_needed = (epsilon_eval * eta_kmax**4) ** (1 / 3)

    mpi.printby0("eta_elav * kmax:", eta_elav * kmax)

    freq_nu4 = 0.5 * (nu_2_needed - params.nu_2) * kmax**2

    mpi.printby0("freq_nu4", freq_nu4)
    mpi.printby0("freq_nu4 / freq_nu2", freq_nu4 / (params.nu_2 * kmax**2))

    params.nu_4 = freq_nu4 / kmax**4

    params.output.sub_directory = "milestone"
    params.output.periods_print.print_stdout = movement.period / 50.0

    periods_save = params.output.periods_save
    periods_save.phys_fields = movement.period / 10.0
    periods_save.spatial_means = movement.period / 1000.0
    periods_save.spect_energy_budg = movement.period / 50.0
    periods_save.spectra = movement.period / 100.0

    if args.only_print_params:
        params._print_as_xml()
        return params, None

    if args.only_init:
        params.output.HAS_TO_SAVE = False

    sim = Simul(params)

    if not args.only_init:
        sim.time_stepping.start()

        mpi.printby0(
            "For a video, run something like:\n\n"
            f"cd {sim.output.path_run}; "
            'ipython -i -c "from fluidsim import load_sim_for_plot as load; '
            "sim=load(); sim.output.phys_fields.animate('vx')\""
        )
    else:
        print(sim.oper.shapeX_loc)
        print(sim.oper.shapeK_loc)

    return params, sim


if __name__ == "__main__":
    args = parser.parse_args()
    mpi.printby0(args)

    params, sim = main(args)
