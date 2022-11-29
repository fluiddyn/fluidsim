"""Simulations with the solver ns3d.strat and the forcing tcrandom_anisotropic.

.. autofunction:: create_parser

.. autofunction:: main

"""

from math import pi, asin, sin
import argparse
import sys

import matplotlib.pyplot as plt

from fluiddyn.util import mpi
from fluidsim.util.scripts import parse_args

doc = """Launcher for simulations with the solver ns3d.strat and
the forcing tcrandom_anisotropic.

Examples
--------

```
./run_simul.py --only-print-params
./run_simul.py --only-print-params-as-code

./run_simul.py -F 0.3 --delta-F 0.1 --ratio-kfmin-kf 0.8 --ratio-kfmax-kf 1.5 -opf
./run_simul.py -F 1.0 --delta-F 0.1 --ratio-kfmin-kf 0.8 --ratio-kfmax-kf 1.5 -opf

mpirun -np 2 ./run_simul.py
```

Notes
-----

This script is designed to study stratified turbulence forced with an
anisotropic forcing in toroidal or poloidal modes.

The regime depends on the value of the horizontal Froude number Fh and buoyancy
Reynolds numbers R and R4:

- Fh = epsK / (Uh^2 N)
- R = epsK / (N^2 nu)
- R4 = epsK Uh^2 / (nu4 N^4)

Fh has to be very small to be in a strongly stratified regime. R and R4 has to
be "large" to be in a turbulent regime.

For this forcing, we fix the injection rate P (very close to epsK). We will
work at P = 1.0, such that N, nu and nu4 determine the non dimensional numbers.

Note that Uh is not directly fixed by the forcing but should be a function of
the other input parameters. Dimensionally, we can write Uh = (P Lfh)**(1/3).

For simplicity, we'd like to have Lfh=1.0. We want to force at "large
horizontal scale" (compared to the size of the numerical domain). This length
(params.oper.Lx = params.oper.Ly) is computed with this condition.

"""

keys_versus_kind = {
    "toro": "vt_fft",
    "polo": "vp_fft",
    "buoyancy": "b_fft",
    "vert": "vz_fft",
}


def create_parser():
    """Create the argument parser with default arguments"""
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # parameters to study the input parameters without running the simulation

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

    # grid parameter

    parser.add_argument(
        "-nz",
        "--nz",
        type=int,
        default=32,
        help="Number of numerical points over one vertical axis",
    )

    # physical parameters

    parser.add_argument(
        "--ratio-nh-nz",
        type=int,
        default=4,
        help=(
            "Ratio nh / nz (has to be a positive integer, "
            "fixes the aspect ratio of the numerical domain)"
        ),
    )

    parser.add_argument("--t_end", type=float, default=10.0, help="End time")

    parser.add_argument(
        "-N", type=float, default=10.0, help="Brunt-Väisälä frequency"
    )

    parser.add_argument("-nu", type=float, default=None, help="Viscosity")

    parser.add_argument(
        "--nu4",
        type=float,
        default=None,
        help="Order-4 hyper viscosity",
    )

    parser.add_argument(
        "-Rb",
        type=float,
        default=None,
        help="Input buoyancy Reynolds number (injection_rate / (nu * N^2))",
    )

    parser.add_argument(
        "--Rb4",
        type=float,
        default=None,
        help="Input order-4 buoyancy Reynolds number (injection_rate / (nu_4 * N^4))",
    )

    parser.add_argument(
        "--coef-nu4",
        type=float,
        default=1.0,
        help="Coefficient used to compute the order-4 hyper viscosity",
    )

    parser.add_argument(
        "--forced-field",
        type=str,
        default="polo",
        help='Forced field (can be "polo", "toro", ...)',
    )

    parser.add_argument(
        "--init-velo-max",
        type=float,
        default=0.01,
        help="params.init_fields.noise.max",
    )

    # shape of the forcing region in spectral space

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

    # Output parameters

    parser.add_argument(
        "--spatiotemporal-spectra",
        action="store_true",
        help="Activate the output spatiotemporal_spectra",
    )

    # Other parameters

    parser.add_argument(
        "--projection",
        type=str,
        default=None,
        help="Type of projection (changes the equations solved)",
    )

    parser.add_argument(
        "--disable-no-vz-kz0",
        action="store_true",
        help="Disable params.no_vz_kz0",
    )

    parser.add_argument(
        "--disable-NO-SHEAR-MODES",
        action="store_true",
        help="Disable params.oper.NO_SHEAR_MODES",
    )

    parser.add_argument(
        "--max-elapsed",
        type=str,
        default="23:50:00",
        help="Max elapsed time",
    )

    parser.add_argument(
        "--coef-dealiasing",
        type=float,
        default=0.8,
        help="params.oper.coef_dealiasing",
    )

    parser.add_argument(
        "--periods_save-phys_fields",
        type=float,
        default=1.0,
        help="params.output.periods_save.phys_fields",
    )

    parser.add_argument(
        "--sub-directory",
        type=str,
        default="aniso",
        help="Sub directory where the simulation will be saved",
    )

    parser.add_argument(
        "--modify-params",
        type=str,
        default=None,
        help="Code modifying the `params` object.",
    )

    return parser


def create_params(args):
    """Create the params object from the script arguments"""

    from fluidsim.solvers.ns3d.strat.solver import Simul

    params = Simul.create_default_params()

    params.output.sub_directory = args.sub_directory
    params.short_name_type_run = args.forced_field

    if args.projection is not None:
        params.projection = args.projection
        params.short_name_type_run += "_proj"

    params.no_vz_kz0 = not args.disable_no_vz_kz0
    params.oper.NO_SHEAR_MODES = not args.disable_NO_SHEAR_MODES
    params.oper.coef_dealiasing = args.coef_dealiasing

    params.oper.truncation_shape = "no_multiple_aliases"

    params.oper.nz = nz = args.nz
    nh = nz * args.ratio_nh_nz

    if args.ratio_nh_nz < 1:
        raise ValueError("args.ratio_nh_nz < 1")

    params.oper.nx = params.oper.ny = nh

    Lfh = 1.0
    injection_rate = 1.0
    Uh = (injection_rate * Lfh) ** (1 / 3)

    Lh = args.nh_forcing * Lfh

    params.oper.Lx = params.oper.Ly = Lh
    params.oper.Lz = Lh / args.ratio_nh_nz
    delta_kz = 2 * pi / params.oper.Lz

    # Brunt Vaisala frequency
    params.N = args.N
    Fh = Uh / (args.N * Lfh)
    params.short_name_type_run += f"_Fh{Fh:.3e}"

    mpi.printby0(f"Input horizontal Froude number: {Fh:.3g}")

    nu = None
    Rb = None
    if args.nu is not None and args.Rb is not None:
        raise ValueError("args.nu is not None and args.Rb is not None")

    if args.nu is not None:
        nu = args.nu
        if nu != 0.0:
            Rb = injection_rate / (nu * args.N**2)
        params.short_name_type_run += f"_nu{nu:.3e}"
        mpi.printby0(f"Input viscosity: {nu:.3e}")
    elif args.Rb is not None:
        Rb = args.Rb
        nu = injection_rate / (Rb * args.N**2)
        params.short_name_type_run += f"_Rb{Rb:.3g}"
        mpi.printby0(f"Input buoyancy Reynolds number: {Rb:.3g}")
    else:
        Rb = 5.0
        nu = injection_rate / (Rb * args.N**2)
    params.nu_2 = nu

    if args.nu4 is not None and args.Rb4 is not None:
        raise ValueError("args.nu4 is not None and args.Rb4 is not None")
    if args.nu4 is not None:
        params.nu_4 = args.nu4
        params.short_name_type_run += f"_nuh{params.nu_4:.3e}"
        mpi.printby0(f"Input order-4 hyper viscosity: {params.nu_4:.3e}")
    elif args.Rb4 is not None:
        Rb_4 = args.Rb4
        params.nu_4 = injection_rate / (Rb_4 * args.N**4)
        params.short_name_type_run += f"_Rbh{Rb_4:.3g}"
        mpi.printby0(f"Input order-4 buoyancy Reynolds number: {Rb_4:.3g}")
    else:
        # compute nu_4 from injection_rate and dx
        # Kolmogorov length scale
        eta = (nu**3 / injection_rate) ** 0.25
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
            Rb_4 = injection_rate / (params.nu_4 * args.N**4)
            mpi.printby0(
                f"Resolution too coarse, we add order-4 hyper viscosity nu_4={params.nu_4:.3e}."
            )

    params.init_fields.type = "noise"
    params.init_fields.noise.length = params.oper.Lz / 2
    params.init_fields.noise.velo_max = args.init_velo_max

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
    params.forcing.nkmax_forcing = min(nz // 2, round3(kf_max / delta_kz))

    mpi.printby0(
        f"{params.forcing.nkmin_forcing = }\n{params.forcing.nkmax_forcing = }"
    )

    period_N = 2 * pi / args.N
    omega_l = args.N * args.F

    # time_stepping fixed to follow waves
    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = args.t_end
    params.time_stepping.max_elapsed = args.max_elapsed
    params.time_stepping.deltat_max = min(0.1, period_N / 16)

    # time_correlation is fixed to forced wave period
    params.forcing.tcrandom.time_correlation = 2 * pi / omega_l
    params.forcing.tcrandom_anisotropic.angle = round3(angle)
    params.forcing.tcrandom_anisotropic.delta_angle = round3(delta_angle)
    params.forcing.tcrandom_anisotropic.kz_negative_enable = True

    params.output.periods_print.print_stdout = 1e-1

    params.output.periods_save.phys_fields = args.periods_save_phys_fields
    params.output.periods_save.spatial_means = 0.02
    params.output.periods_save.spectra = 0.05
    params.output.periods_save.spect_energy_budg = 0.1

    params.output.spectra.kzkh_periodicity = 1

    if args.spatiotemporal_spectra:
        params.output.periods_save.spatiotemporal_spectra = period_N / 8

    params.output.spatiotemporal_spectra.file_max_size = 80.0  # (Mo)
    # probes_region in nondimensional units (mode indices).
    ikzmax = 16
    ikhmax = ikzmax * args.ratio_nh_nz
    params.output.spatiotemporal_spectra.probes_region = (ikhmax, ikhmax, ikzmax)

    if args.modify_params is not None:
        exec(args.modify_params)

    return params


def main(args=None, **defaults):
    """Main function for the scripts based on turb_trandom_anisotropic"""
    parser = create_parser()

    if defaults:
        parser.set_defaults(**defaults)

    args = parse_args(parser, args)

    params = create_params(args)

    if (
        args.only_plot_forcing
        or args.only_print_params_as_code
        or args.only_print_params
    ):
        params.output.HAS_TO_SAVE = False

    sim = None

    if args.only_print_params_as_code:
        params._print_as_code()
        return params, sim

    if args.only_print_params:
        print(params)
        return params, sim

    from fluidsim.solvers.ns3d.strat.solver import Simul

    sim = Simul(params)

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

cd {sim.output.path_run}; fluidsim-ipy-load

# in IPython:

sim.output.phys_fields.set_equation_crosssection('x={params.oper.Lx/2}')
sim.output.phys_fields.animate('b')
"""
    )

    return params, sim


if "sphinx" in sys.modules:
    from textwrap import indent
    from unittest.mock import patch

    with patch.object(sys, "argv", ["run_simul.py"]):
        parser = create_parser()

    __doc__ += """
Example of help message
-----------------------

.. code-block::

""" + indent(
        parser.format_help(), "    "
    )


if __name__ == "__main__":

    params, sim = main()
