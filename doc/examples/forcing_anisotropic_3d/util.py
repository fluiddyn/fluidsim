from math import pi, asin, sin

import argparse

import matplotlib.pyplot as plt

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

    parser.add_argument("--t_end", type=float, default=10.0, help="End time")

    parser.add_argument(
        "-N", type=float, default=10.0, help="Brunt-Väisälä frequency"
    )

    parser.add_argument("-nu", type=float, default=1e-3, help="Viscosity")

    parser.add_argument(
        "--nu4",
        type=float,
        default=None,
        help="Order-4 hyper viscosity",
    )

    parser.add_argument(
        "--coef-nu4",
        type=float,
        default=1.0,
        help="Coefficient used to compute the order-4 hyper viscosity",
    )

    parser.add_argument(
        "-oppac",
        "--only-print-params-as-code",
        action="store_true",
        help="Only run initialization phase and print params as code",
    )

    parser.add_argument(
        "-opp",
        "--only-print-params",
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
        "--ratio-kfmin-kf",
        type=float,
        default=0.5,
        help="",
    )

    parser.add_argument(
        "--ratio-kfmax-kf",
        type=float,
        default=2.0,
        help="",
    )

    parser.add_argument(
        "-F",
        type=float,
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
        params.short_name_type_run += "_proj"

    # TODO: could be parameters of the script
    params.no_vz_kz0 = True
    params.oper.NO_SHEAR_MODES = True
    params.oper.coef_dealiasing = 0.95
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
    params.time_stepping.deltat_max = 0.1

    # Brunt Vaisala frequency
    params.N = args.N
    params.nu_2 = args.nu
    params.short_name_type_run += f"_N{args.N:.3e}"

    if args.nu4 is not None:
        params.nu_4 = args.nu4
    else:
        # compute nu_4 from injection_rate and dx
        # Kolmogorov length scale
        eta = (args.nu ** 3 / injection_rate) ** 0.25
        delta_kz = 2 * pi / params.oper.Lz
        k_max = params.oper.coef_dealiasing * delta_kz * nz / 2

        mpi.printby0(f"{eta * k_max = :.3e}")

        if eta * k_max > 1:
            mpi.printby0("Well resolved simulation, no need for nu_4")
            params.nu_4 = 0.0
        else:
            # TODO: more clever injection_rate_4 (tends to 0 when eta*k_max = 1)
            injection_rate_4 = injection_rate
            # only valid if R4 >> 1 (isotropic turbulence at small scales)
            params.nu_4 = (
                args.coef_nu4 * injection_rate_4 ** (1 / 3) / k_max ** (10 / 3)
            )
            mpi.printby0(f"Resolution too coarse, we add {params.nu_4 = :.3e}.")

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

    angle = asin(args.F)
    mpi.printby0(f"angle = {angle / pi * 180:.2f}°")

    delta_angle = asin(args.delta_F)

    kfh = 2 * pi / Lfh
    kf = kfh / sin(angle)

    kf_min = kf * args.ratio_kfmin_kf
    kf_max = kf * args.ratio_kfmax_kf

    params.forcing.nkmin_forcing = max(0, round3(kf_min / delta_kz))
    params.forcing.nkmax_forcing = min(nz//2, max(1, round3(kf_max / delta_kz)))

    mpi.printby0(f"{params.forcing.nkmin_forcing = }\n{params.forcing.nkmax_forcing = }")

    # time_correlation is fixed to forced wave period
    params.forcing.tcrandom.time_correlation = 2 * pi / (args.N * sin(angle))
    params.forcing.tcrandom_anisotropic.angle = round3(angle)
    params.forcing.tcrandom_anisotropic.delta_angle = round3(delta_angle)
    params.forcing.tcrandom_anisotropic.kz_negative_enable = True

    params.output.periods_print.print_stdout = 1e-1

    params.output.periods_save.phys_fields = 0.5
    params.output.periods_save.spatial_means = 0.02
    params.output.periods_save.spectra = 0.05
    params.output.periods_save.spect_energy_budg = 0.1

    params.output.spectra.kzkh_periodicity = 1
    # TODO: Spatiotemporal aquisition frequency depends on the Brunt Vaisala frenquency (maybe there is a better choice)
    params.output.periods_save.spatiotemporal_spectra = 0.1 / max(1.0, args.N)

    # TODO: Maybe we could implement a smarter probes_region, depending on the number of modes?
    params.output.spatiotemporal_spectra.file_max_size = 80.0
    params.output.spatiotemporal_spectra.probes_region = (20, 20, 10)

    return params


def main():

    args = parse_args()

    params = create_params(args)

    if (
        args.only_plot_forcing
        or args.only_print_params_as_code
        or args.only_print_params
    ):
        params.output.HAS_TO_SAVE = False

    sim = Simul(params)

    if args.only_print_params_as_code:
        params._print_as_code()
        return params, sim

    if args.only_print_params:
        print(params)
        return params, sim

    if args.only_plot_forcing:
        sim.forcing.forcing_maker.plot_forcing_region()
        plt.show()
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

sim.output.phys_fields.set_equation_crosssection('x={params.oper.Lx/2}')
sim.output.phys_fields.animate('b')
"""
    )

    return params, sim


if __name__ == "__main__":

    params, sim = main()
