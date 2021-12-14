"""
To be executed with

```
mpirun -np 2 python run_simul.py -N 96 --Re 5000
```

I guess we need Re~10000 to start to get constant energy fluxes and k^{-5/3}
spectra.

"""

from math import sqrt
import argparse

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.solver import Simul

parser = argparse.ArgumentParser()
parser.add_argument(
    "-N", type=int, default=64, help="Number of grid point along 1 dimension"
)
parser.add_argument("--Re", type=float, default=2000, help="Reynolds number")
parser.add_argument("--t_end", type=float, default=20, help="Nondimension t_end")


def main(args):

    F0 = 1.0
    L = 1.0
    V0 = sqrt(F0 * L)
    T0 = L / V0

    params = Simul.create_default_params()

    params.output.sub_directory = "examples"

    params.nu_2 = sqrt(F0) * L ** (3 / 2) / args.Re

    params.oper.nx = params.oper.ny = params.oper.nz = args.N
    params.oper.Lx = params.oper.Ly = params.oper.Lz = L
    params.oper.coef_dealiasing = 0.75

    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = args.t_end * T0

    params.init_fields.type = "noise"
    params.init_fields.noise.velo_max = 0.001

    params.forcing.enable = True
    params.forcing.type = "taylor_green"
    params.forcing.taylor_green.amplitude = F0

    params.output.periods_print.print_stdout = 1.0
    params.output.periods_save.phys_fields = 0.5
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spatial_means = 0.1
    params.output.periods_save.spect_energy_budg = 0.5

    sim = Simul(params)
    sim.time_stepping.start()

    mpi.printby0(
        "# To reload the simul and plot the results:",
        f"cd {sim.output.path_run}",
        'ipython --matplotlib -i -c "from fluidsim import load; sim = load()"',
        "# In IPython:",
        "sim.output.spatial_means.plot()",
        "sim.output.spectra.plot1d(tmin=16, coef_compensate=5/3)",
        "sim.output.phys_fields.plot()",
        "sim.output.spect_energy_budg.plot_fluxes(tmin=16)",
        "sim.output.print_stdout.plot_clock_times()",
        sep="\n",
    )

    return params, sim


if __name__ == "__main__":
    args = parser.parse_args()
    mpi.printby0(args)

    params, sim = main(args)
