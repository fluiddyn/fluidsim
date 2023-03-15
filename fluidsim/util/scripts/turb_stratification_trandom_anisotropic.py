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

doc = """Launcher for simulations with the solver ns3d.strat with stratification and
the forcing tcrandom_anisotropic.

Examples
--------

```
./run_simul.py --only-print-params
./run_simul.py --only-print-params-as-code

./run_simul.py -F 0.3 --delta_F 0.1 --nkmin_forcing 3 --nkmax_forcing 5 -opf

mpirun -np 2 ./run_simul.py
```

Notes
-----

This script is designed to study stratified turbulence forced with an
anisotropic forcing in poloidal or buoyancy modes.

The regime depends on the value of the Froude number Fh and Reynolds numbers Re:

- Fh = U / (N L)
- Re = U L / nu

where U is the rms of the velocity, L the size of the box, N the Brunt-Väisälä frequency and nu is the viscosity.

Fh has to be very small to be in a strong rotation regime and Re has to
be large to be in a turbulent regime.

For this forcing, we fix the injection rate P=1.0 (equal to epsK in the steady state) and we vary N and nu

Note that U is not directly fixed by the forcing but should be a function of
the other input parameters. Dimensionally, if the flow is fully turbulent, we can write U = (P L)**(1/3).

The flow is forced at large spatial scales (compared to the size of the numerical domain) and at specified frequency omega=N*F. 

"""

keys_versus_kind = {
    "polo": "vp_fft",
    "buoyancy": "b_fft"
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
        "-n",
        "--n",
        type=int,
        default=64,
        help="Number of numerical points over one axis",
    )

    # physical parameters

    parser.add_argument("--t_end", type=float, default=10.0, help="End time")

    parser.add_argument(
        "-N", type=float, default=None, help="Brunt-Väisälä frequency"
    )
    
    parser.add_argument(
        "--Fh",
        type=float,
        default=None,
        help="Input Froude number",
    )
    
    parser.add_argument(
        "-coef_nu", 
        type=float, 
        default=1.2, 
        help="Coefficient used to compute the viscosity. It should correspond to kmax*eta"
    )
    
    parser.add_argument(
        "--Re",
        type=float,
        default=None,
        help="Input Reynolds number",
    )

    parser.add_argument(
        "--forced-field",
        type=str,
        default="polo",
        help='Forced field (can be "polo", "toro", or "vert")',
    )

    # shape of the forcing region in spectral space

    parser.add_argument(
        "--nkmin_forcing",
        type=int,
        default=3,
        help="Minimal dimensionless wavenumber forced",
    )

    parser.add_argument(
        "--nkmax_forcing",
        type=int,
        default=5,
        help="Maximal dimensionless wavenumber forced",
    )

    parser.add_argument(
        "-F",
        type=float,
        default=0.97,
        help="Ratio omega_f / N, fixing the mean angle between the vertical and the forced wavenumber",
    )

    parser.add_argument(
        "--delta_F",
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
        "--NO_SHEAR_MODES",
        type=bool,
        default=False,
        help="params.oper.NO_SHEAR_MODES",
    )

    parser.add_argument(
        "--max-elapsed",
        type=str,
        default="19:45:00",
        help="Max elapsed time",
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
        default="aniso_stratification",
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

    params.no_vz_kz0 = False
    params.oper.NO_SHEAR_MODES = args.NO_SHEAR_MODES
    params.oper.NO_GEOSTROPHIC_MODES = False
    if params.oper.NO_SHEAR_MODES:
        params.short_name_type_run += f"_NO_SHEAR_MODES"
    params.oper.coef_dealiasing = coef_dealiasing = 2./3.
    params.oper.truncation_shape = "no_multiple_aliases"
    params.oper.nx = params.oper.ny = params.oper.nz = n = args.n
    params.oper.Lx = params.oper.Ly = params.oper.Lz = L = 2*pi
    
    delta_k = 2 * pi / L
    k_max = coef_dealiasing * delta_k * n / 2
    injection_rate = 1.0
    U = (injection_rate * L) ** (1 / 3)

    
    # Brunt-Väisälä frequency N and Froude number
    if args.N is not None and args.Fh is not None:
        raise ValueError("args.N is not None and args.Fh is not None")
    if args.N is not None:
        N = args.N
        mpi.printby0(f"Input Brunt-Väisälä frequency: {N:.3g}")
        if N != 0.0:
            Fh = U / (N * L)
            params.short_name_type_run += f"_Fh{Fh:.3e}"
        else:
            params.short_name_type_run += f"_N{N:.3e}"
    elif args.Fh is not None:
        Fh = args.Fh
        mpi.printby0(f"Input Froude number: {Fh:.3g}")
        if Fh != 0.0:
            N = U / (Fh * L)
            params.short_name_type_run += f"_Fh{Fh:.3e}"
        else:
            raise ValueError("Fh = 0.0")
    else:
        raise ValueError("args.N is None and args.Fh is None")
    params.N = N
   
   
    # Viscosity and Reynolds number
    if args.coef_nu is not None and args.Re is not None:
        raise ValueError("args.coef_nu is not None and args.Re is not None")
    if args.coef_nu is not None:
        coef_nu = args.coef_nu
        mpi.printby0(f"Input coefficient for viscosity: {coef_nu:.3g}")
        if coef_nu != 0.0:
            # only valid if R >> 1 (isotropic turbulence at small scales)
            nu_eddies = (
                injection_rate ** (1 / 3) * (coef_nu / k_max) ** (4 / 3)
            )
            # dissipation frequency = maximal wave frequency
            nu_waves = (
                N * (coef_nu / k_max) ** 2
            )
            if nu_waves > nu_eddies:
                print("Viscosity fixed by waves")
            nu = max(nu_eddies, nu_waves)
        else:
            nu = 0.0
    elif args.Re is not None:
        Re = args.Re
        mpi.printby0(f"Input Reynolds number: {Re:.3g}")
        if Re != 0.0:
            nu = U * L / Re
        else:
            raise ValueError("Re = 0.0")
    else:
        raise ValueError("args.coef_nu is None and args.Re is None")
    params.nu_2 = nu
    
     

    params.init_fields.type = "noise"
    params.init_fields.noise.length = L / 2
    params.init_fields.noise.velo_max = 0.01
    
    period_N = 2 * pi / N
    omega_f = N * args.F

    # time_stepping fixed to follow waves
    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = args.t_end
    params.time_stepping.max_elapsed = args.max_elapsed
    params.time_stepping.deltat_max = min(0.02, period_N * Fh / 200)
    params.time_stepping.USE_CFL = True
    params.time_stepping.cfl_coef = 0.1
    params.time_stepping.type_time_scheme = "RK4"

    # forcing
    params.forcing.enable = True
    params.forcing.type = "tcrandom_anisotropic"
    params.forcing.forcing_rate = injection_rate
    params.forcing.key_forced = keys_versus_kind[args.forced_field]


    params.forcing.nkmin_forcing = delta_k * args.nkmin_forcing
    params.forcing.nkmax_forcing = delta_k * args.nkmax_forcing
    
    angle = asin(args.F)
    delta_angle = asin(args.delta_F)    
    mpi.printby0(
        f"{params.forcing.nkmin_forcing = }\n{params.forcing.nkmax_forcing = }"
    )
    mpi.printby0(f"angle = {angle / pi * 180:.2f}°")
    mpi.printby0(f"delta_angle = {delta_angle / pi * 180:.2f}°")
    #  time_correlation is fixed to forced wave period
    params.forcing.tcrandom.time_correlation = 2 * pi / omega_f
    params.forcing.tcrandom_anisotropic.angle = angle
    params.forcing.tcrandom_anisotropic.delta_angle = delta_angle
    params.forcing.tcrandom_anisotropic.kz_negative_enable = True
    

    params.output.periods_print.print_stdout = 1e-1

    params.output.periods_save.phys_fields = args.periods_save_phys_fields
    params.output.periods_save.spatial_means = min(0.1, period_N / 50)
    params.output.periods_save.spectra = 0.05
    params.output.periods_save.spect_energy_budg = 0.1

    params.output.spectra.kzkh_periodicity = 1

    if args.spatiotemporal_spectra:
        params.output.periods_save.spatiotemporal_spectra = period_N / 8

    params.output.spatiotemporal_spectra.file_max_size = 80.0  # (Mo)
    # probes_region in nondimensional units (mode indices).
    ikmax = 16
    params.output.spatiotemporal_spectra.probes_region = (ikmax, ikmax, ikmax)

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
