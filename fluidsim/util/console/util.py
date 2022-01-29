"""Utilities for benchmarking and profiling (:mod:`fluidsim.util.bench.util`)
=============================================================================

"""
import os
from time import time
from pathlib import Path

try:
    from time import perf_counter as clock
except ImportError:
    # python 2.7
    from time import clock
import json
import socket
import shutil

from fluiddyn.util import time_as_str
from fluiddyn.util import mpi
from fluiddyn.io import stdout_redirected
from ..util import available_solver_keys, get_dim_from_solver_key


old_print = print
print = mpi.printby0


class ConsoleError(ValueError):
    """Distinguish errors from console utilities."""

    pass


def modif_box_size(params, n0, n1, n2=None):
    """Modify box size, such that the aspect ratio is square / cube

    Parameters
    ----------
    params : ParamContainer
        Input parameters
    n0 : int
        Number of grid points in 0-axis of the grid
    n1 : int
        Number of grid points in 1-axis of the grid
    n2 :
        Number of grid points in 2-axis of the grid

    Returns
    -------

    """
    if n2 is None:
        nx = n1
        ny = n0
    else:
        nx = n2
        ny = n1
        nz = n0

    if nx != ny:
        if nx < ny:
            params.oper.Ly = params.oper.Ly * ny / nx
        else:
            params.oper.Lx = params.oper.Lx * nx / ny

    if n2 is None:
        mpi.printby0(
            "nh = ({}, {}); Lh = ({}, {})".format(
                n0, n1, params.oper.Ly, params.oper.Lx
            )
        )

    if n2 is not None and n2 != n0:
        params.oper.Lz = params.oper.Lx * nz / nx

        mpi.printby0(
            "n = ({}, {}, {}); L = ({}, {}, {})".format(
                n0, n1, n2, params.oper.Lz, params.oper.Ly, params.oper.Lx
            )
        )


def modif_params2d(
    params, n0=3 * 2**8, n1=None, name_run="profile", type_fft=None, it_end=20
):
    """Modify parameters for 2D benchmarks.

    Parameters
    ----------
    params : ParamContainer
        Default parameters
    n0 : int
        Number of grid points in x-axis of the grid
    n1 : int
        Number of grid points in y-axis of the grid
    name_run : str
        Short name of the run
    type_fft : str
        When set, uses a diffrent FFT type than the default

    """
    params.short_name_type_run = name_run

    if n1 is None:
        n1 = n0

    params.oper.nx = n1
    params.oper.ny = n0
    modif_box_size(params, n0, n1)

    if type_fft is not None:
        params.oper.type_fft = type_fft

    if "FLUIDSIM_NO_FLUIDFFT" not in os.environ:
        # params.oper.type_fft = 'fft2d.mpi_with_fftwmpi2d'
        pass

    if "noise" in params.init_fields.available_types:
        params.init_fields.type = "noise"

    # params.forcing.enable = True
    # params.forcing.type = "tcrandom"
    # params.forcing.nkmax_forcing = 6
    # params.forcing.nkmin_forcing = 3
    # params.forcing.forcing_rate = 1.

    params.nu_8 = 1.0
    try:
        params.f = 1.0
        params.c2 = 200.0
    except AttributeError:
        pass

    try:
        params.N = 1.0
    except AttributeError:
        pass

    params.time_stepping.deltat0 = 1.0e-6
    params.time_stepping.USE_CFL = False
    params.time_stepping.it_end = it_end
    params.time_stepping.USE_T_END = False

    params.output.periods_print.print_stdout = 0
    params.output.HAS_TO_SAVE = 0


def modif_params3d(
    params, n0=256, n1=None, n2=None, name_run="profile", type_fft=None, it_end=10
):
    """Modify parameters for 3D benchmarks.

    Parameters
    ----------
    params : ParamContainer
        Default parameters
    n0 : int
        Number of grid points in x-axis of the grid
    n1 : int
        Number of grid points in y-axis of the grid
    n2 : int
        Number of grid points in z-axis of the grid
    name_run : str
        Short name of the run
    type_fft : str
        When set, uses a diffrent FFT type than the default

    """
    params.short_name_type_run = name_run

    if n1 is None:
        n1 = n0

    if n2 is None:
        n2 = n0

    params.oper.nx = n2
    params.oper.ny = n1
    params.oper.nz = n0
    modif_box_size(params, n0, n1, n2)

    if type_fft is not None:
        params.oper.type_fft = type_fft

    if "FLUIDSIM_NO_FLUIDFFT" not in os.environ:
        # params.oper.type_fft = 'fft2d.mpi_with_fftwmpi2d'
        pass

    # params.forcing.enable = False
    # params.forcing.type = 'tcrandom'
    # params.forcing.nkmax_forcing = 5
    # params.forcing.nkmin_forcing = 4
    # params.forcing.forcing_rate = 1.

    params.nu_8 = 1.0
    try:
        params.f = 1.0
        params.c2 = 200.0
    except AttributeError:
        pass

    try:
        params.N = 1.0
    except AttributeError:
        pass

    if "noise" in params.init_fields.available_types:
        params.init_fields.type = "noise"

    params.time_stepping.deltat0 = 1.0e-4
    params.time_stepping.USE_CFL = False
    params.time_stepping.it_end = it_end
    params.time_stepping.USE_T_END = False

    params.output.periods_print.print_stdout = 0
    params.output.HAS_TO_SAVE = 0


def init_parser_base(parser):
    """Initalize argument parser with common arguments for benchmarking
    analysis and profile console tools.

    Parameters
    ----------
    parser : argparse.ArgumentParser

    """
    parser.add_argument("n0", nargs="?", type=int, default=None)
    parser.add_argument("n1", nargs="?", type=int, default=None)
    parser.add_argument("n2", nargs="?", type=int, default=None)

    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default=None,
        help="Any of the following solver keys: {}".format(
            available_solver_keys()
        ),
    )
    parser.add_argument("-d", "--dim", default=None)


def parse_args_dim(args):
    """Parse dimension argument args.dim

    Parameters
    ----------
    args : argparse.ArgumentParser

    """
    dim = args.dim
    n0 = args.n0
    n1 = args.n1
    n2 = args.n2
    solver = args.solver

    if solver is not None:
        dim = get_dim_from_solver_key(solver)

    if dim is None:
        if n0 is not None and n1 is not None and n2 is None:
            dim = "2d"
        elif n0 is not None and n1 is not None and n2 is not None:
            dim = "3d"
        else:
            raise ConsoleError(
                "Cannot determine which shape you want to use for this bench "
                "('2d' or '3d')"
            )

    if dim.lower() in ["3", "3d"]:
        if n0 is None:
            n0 = 128
        if n2 is None:
            n2 = n0
        dim = "3d"
    elif dim.lower() in ["2", "2d"]:
        dim = "2d"
        if n0 is None:
            n0 = 512
    else:
        raise ConsoleError(f"dim should not be {dim}")

    if n1 is None:
        n1 = n0

    if solver is None:
        solver = "ns2d" if dim == "2d" else "ns3d"

    args.dim = dim
    args.n0 = n0
    args.n1 = n1
    args.n2 = n2
    args.solver = solver

    return args


def get_path_file(sim, path_dir_results, name="bench", ext=".json"):
    """Generate a unique filename from simulation object."""

    path_dir_results = Path(path_dir_results)

    if mpi.rank == 0:
        path_dir_results.mkdir(parents=True, exist_ok=True)

    t_as_str = time_as_str()
    key_solver = sim.info_solver.short_name.lower()
    pid = str(os.getpid())
    nb_proc = f"np={mpi.nb_proc}"
    type_fft = sim.params.oper.type_fft.split(".")[-1].replace("_", "-")
    name_file = (
        "_".join(
            [
                "result",
                name,
                key_solver,
                sim.oper.produce_str_describing_grid(),
                nb_proc,
                type_fft,
                t_as_str + pid,
            ]
        )
        + ext
    )

    path = str(path_dir_results / name_file)
    return path, t_as_str


def bench(sim, path_dir_results):
    """Benchmark a simulation run and save the results in a JSON file.

    Parameters
    ----------
    sim : Simul
        An initialized simulation object
    path_dir_results :  str
        Directory path to save results in

    """
    path, t_as_str = get_path_file(sim, path_dir_results)
    print("running a benchmark simulation... ", end="")
    with stdout_redirected():
        t0_usr = time()
        t0_sys = clock()
        sim.time_stepping.start()
        t_elapsed_sys = clock() - t0_sys
        t_elapsed_usr = time() - t0_usr

    print(
        "done.\n{} time steps computed in {:.2f} s".format(
            sim.time_stepping.it, t_elapsed_usr
        )
    )

    if sim.oper.rank != 0:
        return

    results = {
        "t_elapsed_usr": t_elapsed_usr,
        "t_elapsed_sys": t_elapsed_sys,
        "key_solver": sim.info_solver.short_name.lower(),
        "n0": sim.oper.shapeX_seq[0],
        "n1": sim.oper.shapeX_seq[1],
        "n0_loc": sim.oper.shapeX_loc[0],
        "n1_loc": sim.oper.shapeX_loc[1],
        "k0_loc": sim.oper.shapeK_loc[0],
        "k1_loc": sim.oper.shapeK_loc[1],
        "nb_proc": mpi.nb_proc,
        "pid": os.getpid(),
        "time_as_str": t_as_str,
        "hostname": socket.gethostname(),
        "nb_iter": sim.params.time_stepping.it_end,
        "type_fft": sim.oper.type_fft,
    }

    try:
        results.update(
            {
                "n2": sim.oper.shapeX_seq[2],
                "n2_loc": sim.oper.shapeX_loc[2],
                "k2_loc": sim.oper.shapeK_loc[2],
            }
        )
    except IndexError:
        pass

    with open(path, "w") as file:
        json.dump(results, file, sort_keys=True)
        file.write("\n")

    print(f"results benchmarks saved in\n{path}\n")


def tear_down(sim):
    """Delete simulation directory."""
    if mpi.rank == 0:
        print("Cleaning up simulation.")
        shutil.rmtree(sim.output.path_run, ignore_errors=True)
