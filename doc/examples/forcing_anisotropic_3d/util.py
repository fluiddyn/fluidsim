from math import pi, tan, asin

import argparse

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.strat.solver import Simul

keys_versus_kind = {
    "toro": "vt_fft",
    "polo": "vp_fft",
    "buoyancy": "b_fft",
    "vert": "vz_fft",
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--t_end", type=float, default=50.0, help="End time")

    parser.add_argument(
        "-N", type=float, default=2.0, help="Brunt-Väisälä frequency"
    )

    parser.add_argument("-nu", type=float, default=1e-3, help="Viscosity")

    parser.add_argument(
        "--coef-nu4",
        type=float,
        default=1.0,
        help="Coefficient used to compute the order-4 hyper viscosity",
    )

    parser.add_argument(
        "-oi",
        "--only_init",
        action="store_true",
        help="Only run initialization phase and print params",
    )

    parser.add_argument(
        "-opf",
        "--only-plot-forcing",
        action="store_true",
        help="Only run initialization phase and plot forcing",
    )

    parser.add_argument(
        "-nz",
        "--nz",
        type=int,
        default=32,
        help="Number of numerical points over one horizontal axis",
    )

    parser.add_argument(
        "--ratio-nh-nz",
        type=int,
        default=4,
        help="Ratio nh / nz (has to be a positive integer)",
    )

    parser.add_argument(
        "--nh-forcing",
        type=int,
        default=3,
        help="Dimensionless horizontal wavenumber forced",
    )

    parser.add_argument(
        "--delta-nz-forcing",
        type=int,
        default=3,
        help="... not nice... physics change with ratio-nh-nz",
    )

    parser.add_argument(
        "-F",
        default=0.3,
        help="Ratio omega_f / N, fixing the mean angle between the vertical and the forced wavenumber",
    )

    parser.add_argument(
        "--delta-F",
        type=float,
        default=0.1,
        help="delta F, fixing angle_max - angle_min",
    )

    parser.add_argument(
        "--forced-field",
        type=str,
        default="polo",
        help='Forced field (can be "polo", "toro", ...)',
    )

    parser.add_argument(
        "--projection",
        type=str,
        default=None,
        help="Type of projection",
    )

    parser.add_argument(
        "--max-elapsed",
        type=str,
        default="23:50:00",
        help="Max elapsed time",
    )

    args = parser.parse_args()
    mpi.printby0(args)

    return args


def create_params(args):

    params = Simul.create_default_params()

    params.output.sub_directory = "aniso"
    params.short_name_type_run = args.forced_field

    if args.projection is not None:
        params.projection = args.projection
        # TODO: change short_name_type_run

    params.no_vz_kz0 = True
    params.oper.NO_SHEAR_MODES = True

    params.oper.coef_dealiasing = 0.8
    params.oper.truncation_shape = "no_multiple_aliases"

    params.oper.nz = nz = args.nz
    nh = nz * args.ratio_nh_nz

    # TODO: more checks on the value args.ratio_nh_nz
    if args.ratio_nh_nz < 1:
        raise ValueError

    params.oper.nx = params.oper.ny = nh

    Lfh = 1.0
    injection_rate = 1.0

    Lh = args.nh_forcing * Lfh

    params.oper.Lx = params.oper.Ly = Lh
    params.oper.Lz = Lh / args.ratio_nh_nz

    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = args.t_end
    params.time_stepping.max_elapsed = args.max_elapsed

    # Brunt Vaisala frequency
    params.N = args.N
    params.nu_2 = args.nu

    # TODO: compute nu_4 from injection_rate and dx
    args.coef_nu4
    # Kolmogorov length scale
    eta = (args.nu ** 3 / injection_rate) ** 0.25
    delta_kz = 2 * pi / params.oper.Lz
    k_max = params.oper.coef_dealiasing * delta_kz * nz / 2

    print(f"{eta * k_max = :.3e}")

    if eta * k_max > 1:
        print("Well resolved simulation, no need for nu_4")
        params.nu_4 = 0.0
    else:
        ...

    params.init_fields.type = "noise"
    params.init_fields.noise.length = 1.0
    params.init_fields.noise.velo_max = 0.01

    params.forcing.enable = True
    params.forcing.type = "tcrandom_anisotropic"
    params.forcing.forcing_rate = injection_rate
    params.forcing.key_forced = keys_versus_kind[args.forced_field]

    """
    Since args.ratio_nh_nz > 1,

    - kf_min = delta_kz * nkmin_forcing
    - kf_max = delta_kz * nkmax_forcing

    """

    def round3(number):
        return round(number, 3)

    angle = round3(asin(args.F))
    delta_angle = round3(asin(args.delta_F))

    kfh = 2 * pi / Lfh
    kfz = kfh * tan(angle)
    nkz_forcing = kfz / delta_kz

    params.forcing.nkmin_forcing = round3(
        max(1, nkz_forcing - args.delta_nz_forcing)
    )
    params.forcing.nkmax_forcing = round3(nkz_forcing + args.delta_nz_forcing)

    # TODO: compute time_correlation from forced wave time
    params.forcing.tcrandom.time_correlation = 1.0
    params.forcing.tcrandom_anisotropic.angle = angle
    params.forcing.tcrandom_anisotropic.delta_angle = delta_angle
    params.forcing.tcrandom_anisotropic.kz_negative_enable = True

    params.output.periods_print.print_stdout = 1e-1

    params.output.periods_save.phys_fields = 0.5
    params.output.periods_save.spatial_means = 0.1
    params.output.periods_save.spectra = 0.1
    params.output.periods_save.spect_energy_budg = 0.1
    params.output.spectra.kzkh_periodicity = 1

    # TODO: params.output.spatiotemporal_spectra

    return params


def main():

    args = parse_args()

    params = create_params(args)

    if args.only_plot_forcing or args.only_init:
        params.output.HAS_TO_SAVE = False

    sim = Simul(params)

    if args.only_init:
        params._print_as_code()
        return params, sim

    if args.only_plot_forcing:
        sim.forcing.forcing_maker.plot_forcing_region()
        return params, sim

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

    return params, sim


if __name__ == "__main__":

    params, sim = main()
