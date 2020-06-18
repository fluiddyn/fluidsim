"""Taylor-Green Vortex at Re = 1600
===================================

https://www.grc.nasa.gov/hiocfd/wp-content/uploads/sites/22/case_c3.3.pdf


Example:

```
python run_simul.py -nx 64 -cd 0.9 --type_time_scheme "RK2" -cfl 0.2 --max-elapsed "00:03:00"
```

"""

import argparse
from pathlib import Path
import sys

import numpy as np

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.solver import Simul
from fluidsim import FLUIDSIM_PATH, load_for_restart

sub_directory = "taylor-green_phase-shifting"

parser = argparse.ArgumentParser()
parser.add_argument("-nx", type=int, default=96, help="Resolution")

parser.add_argument(
    "-cd",
    "--coef-dealiasing",
    type=float,
    default=2 / 3,
    help="Coefficient dealiasing",
)

parser.add_argument(
    "-cfl", "--cfl-coef", type=float, default=None, help="CFL coefficient"
)

parser.add_argument("--type_time_scheme", type=str, default="RK4")

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
    help='Maximum elapsed time (for example "02:00:00")',
    type=str,
    default=None,
)

parser.add_argument(
    "--t_end", help="params.time_stepping.t_end", type=float, default=20.0,
)

parser.add_argument(
    "--nb_pairs",
    help="params.time_stepping.phaseshift_random.nb_pairs",
    type=int,
    default=1,
)

parser.add_argument(
    "--nb_steps_compute_new_pair",
    help="params.time_stepping.phaseshift_random.nb_steps_compute_new_pair",
    type=int,
    default=None,
)


Re = 1600
V0 = 1.0
L = 1


def init_params(args):

    params = Simul.create_default_params()

    params.short_name_type_run = (
        f"{args.type_time_scheme}_trunc{args.coef_dealiasing:.3f}"
    )

    if args.nb_pairs != 1:
        params.short_name_type_run += f"_nb_pairs{args.nb_pairs}"

    if args.nb_steps_compute_new_pair:
        params.short_name_type_run += f"_nb_steps{args.nb_steps_compute_new_pair}"

    if args.cfl_coef:
        params.short_name_type_run += f"_cfl{args.cfl_coef}"

    params.nu_2 = V0 * L / Re

    params.init_fields.type = "in_script"

    params.time_stepping.t_end = args.t_end * L / V0
    params.time_stepping.type_time_scheme = args.type_time_scheme
    if args.max_elapsed is not None:
        params.time_stepping.max_elapsed = args.max_elapsed
    params.time_stepping.cfl_coef = args.cfl_coef

    params_phaseshift = params.time_stepping.phaseshift_random
    params_phaseshift.nb_pairs = args.nb_pairs
    params_phaseshift.nb_steps_compute_new_pair = args.nb_steps_compute_new_pair

    params.oper.nx = params.oper.ny = params.oper.nz = args.nx
    params.oper.Lx = params.oper.Ly = params.oper.Lz = 2 * np.pi * L
    params.oper.coef_dealiasing = args.coef_dealiasing
    if args.coef_dealiasing > 2 / 3:
        params.oper.truncation_shape = "no_multiple_aliases"

    params.output.sub_directory = sub_directory
    params.output.periods_print.print_stdout = 0.5
    params.output.periods_save.phys_fields = 4
    params.output.periods_save.spatial_means = 0.2
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.spectra.kzkh_periodicity = 1

    if args.only_init:
        params.output.HAS_TO_SAVE = False
        params.NEW_DIR_RESULTS = False

    return params


def init_state(sim):
    X, Y, Z = sim.oper.get_XYZ_loc()

    vx = V0 * np.sin(X / L) * np.cos(Y / L) * np.cos(Z / L)
    vy = -V0 * np.cos(X / L) * np.sin(Y / L) * np.cos(Z / L)
    vz = sim.oper.create_arrayX(value=0)

    sim.state.init_statephys_from(vx=vx, vy=vy, vz=vz)

    sim.state.statespect_from_statephys()
    sim.state.statephys_from_statespect()


def init_new_simul(args):

    params = init_params(args)

    if args.only_print_params:
        params.time_stepping._print_as_xml()
        params.oper._print_as_xml()
        params.output._print_as_xml()
        sys.exit()

    sim = Simul(params)
    init_state(sim)
    return params, sim


if __name__ == "__main__":
    args = parser.parse_args()
    mpi.printby0(args)

    path_dir = Path(FLUIDSIM_PATH) / sub_directory

    name_file = (
        f"idempotent_nx{args.nx}_"
        f"{args.type_time_scheme}_trunc{args.coef_dealiasing:.3f}"
    )

    if args.cfl_coef:
        name_file += f"_cfl{args.cfl_coef}"

    if args.nb_pairs != 1:
        name_file += f"_nb_pairs{args.nb_pairs}"

    if args.nb_steps_compute_new_pair:
        name_file += f"_nb_steps{args.nb_steps_compute_new_pair}"

    name_file += ".txt"

    path_idempotent_file = path_dir / name_file

    path_idempotent_file_exists = None
    if mpi.rank == 0:
        path_idempotent_file_exists = path_idempotent_file.exists()
    if mpi.nb_proc > 1:
        path_idempotent_file_exists = mpi.comm.bcast(
            path_idempotent_file_exists, root=0
        )

    if not path_idempotent_file_exists or args.only_print_params:
        params, sim = init_new_simul(args)
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

        params, _ = load_for_restart(path_dir / name_dir)
        sim = Simul(params)

    if not args.only_init:
        sim.time_stepping.start()
    else:
        print(sim.oper.shapeX_loc)
        print(sim.oper.shapeK_loc)
        sys.exit()

    if sim.time_stepping._has_to_stop:
        mpi.printby0(
            "Simulation is not completed. The script should be relaunched"
        )
        sys.exit(99)

    mpi.printby0(
        f"""
To visualize the output with Paraview, create a file states_phys.xmf with:

fluidsim-create-xml-description {sim.output.path_run}

# To visualize with fluidsim:

cd {sim.output.path_run}
ipython --matplotlib

# in ipython:

from fluidsim import load_sim_for_plot as load
sim = load()

sim.output.spatial_means.plot()
sim.output.spectra.plot1d(tmin=12, tmax=16, coef_compensate=5/3)

sim.output.phys_fields.set_equation_crosssection(f'x={{sim.oper.Lx/4}}')
sim.output.phys_fields.plot(field="vx", time=10)

sim.output.phys_fields.animate('vx')

"""
    )
