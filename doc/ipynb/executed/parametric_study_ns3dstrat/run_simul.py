from math import pi
import argparse

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.strat.solver import Simul
from fluidsim.base.forcing.milestone import PeriodicUniform

from fluidsim.extend_simul.spatial_means_regions_milestone import (
    SpatialMeansRegions,
)
from fluidsim.extend_simul import extend_simul_class

Simul = extend_simul_class(Simul, SpatialMeansRegions)

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=float, default=0.5, help="Brunt–Väisälä frequency")

parser.add_argument(
    "-D", "--diameter", type=float, default=0.25, help="Diameter of the cylinders"
)

parser.add_argument(
    "-s",
    "--speed",
    type=float,
    default=0.1,
    help="Maximum speed of the cylinders",
)

parser.add_argument(
    "-nc", "--number_cylinders", type=int, default=3, help="Number of cylinders"
)

parser.add_argument(
    "-nypc",
    "--ny_per_cylinder",
    type=int,
    default=60,
    help="Number of numerical points for one cylinder",
)

parser.add_argument(
    "-np",
    "--n_periods",
    type=float,
    default=5,
    help="Number of periods",
)

parser.add_argument(
    "-oi",
    "--only_init",
    action="store_true",
    help="Only run initialization phase",
)

parser.add_argument(
    "-wbl",
    "--width_boundary_layers",
    type=float,
    default=0.02,
    help="width boundary layers",
)


def main(args):

    params = Simul.create_default_params()

    diameter = args.diameter  # m
    speed = args.speed  # m/s
    params.N = args.N  # rad/s

    mesh = 3 * diameter
    number_cylinders = args.number_cylinders

    ly = params.oper.Ly = mesh * number_cylinders
    lx = params.oper.Lx = 1.5 * 3
    ny = params.oper.ny = args.ny_per_cylinder * number_cylinders

    nx_float = ny * lx / ly
    nx = params.oper.nx = round(nx_float)
    assert nx == nx_float

    dx = lx / nx
    mpi.printby0(f"{dx = }")

    lz = params.oper.Lz = mesh
    params.oper.nz = round(lz / dx)

    params.oper.coef_dealiasing = 0.8
    params.oper.NO_SHEAR_MODES = True
    params.no_vz_kz0 = True

    params.forcing.enable = True
    params.forcing.type = "milestone"
    params.forcing.milestone.nx_max = min(
        nx, round(16 * number_cylinders * nx / ny)
    )
    mpi.printby0(f"{params.forcing.milestone.nx_max = }")

    objects = params.forcing.milestone.objects

    objects.number = number_cylinders
    objects.diameter = diameter
    objects.width_boundary_layers = args.width_boundary_layers
    assert objects.width_boundary_layers < diameter / 4

    movement = params.forcing.milestone.movement

    movement.type = "periodic_uniform"
    movement.periodic_uniform.length = lx - 2 * diameter
    movement.periodic_uniform.length_acc = 0.25
    movement.periodic_uniform.speed = speed

    params.init_fields.type = "noise"
    params.init_fields.noise.velo_max = 5e-3

    movement = PeriodicUniform(
        speed,
        movement.periodic_uniform.length,
        movement.periodic_uniform.length_acc,
        lx,
    )

    params.time_stepping.t_end = movement.period * args.n_periods
    params.time_stepping.deltat_max = 0.04 * diameter / speed
    mpi.printby0(f"{params.time_stepping.deltat_max = }")

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

    params.output.sub_directory = "tutorial_parametric_study"
    if nx > 500:
        params.output.periods_print.print_stdout = movement.period / 50.0
    else:
        params.output.periods_print.print_stdout = movement.period / 20.0

    periods_save = params.output.periods_save
    periods_save.phys_fields = movement.period / 10.0
    periods_save.spatial_means = movement.period / 1000.0
    periods_save.spatial_means_regions = movement.period / 1000.0
    periods_save.spect_energy_budg = movement.period / 50.0
    periods_save.spectra = movement.period / 100.0
    periods_save.spatiotemporal_spectra = 2 * pi / params.N / 4

    params.output.spatial_means_regions.xmin = [0, 0.1, 0.4, 0.7]
    params.output.spatial_means_regions.xmax = [1, 0.3, 0.6, 0.9]

    spatiotemporal_spectra = params.output.spatiotemporal_spectra
    spatiotemporal_spectra.file_max_size = 20.0
    spatiotemporal_spectra.probes_region = (10 * lx / ly, 10, 10)

    params.output.spectra.kzkh_periodicity = 2

    sim = Simul(params)

    if not args.only_init:
        sim.time_stepping.start()

    return params, sim


if __name__ == "__main__":
    args = parser.parse_args()
    mpi.printby0(args)

    params, sim = main(args)
