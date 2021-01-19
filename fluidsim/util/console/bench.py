"""Run benchmarks (:mod:`fluidsim.util.console.bench`)
======================================================

"""

import gc
from collections import OrderedDict

import numpy as np

from fluiddyn.util import mpi, info
from fluiddyn.io import stdout_redirected

from fluidsim import _is_testing

from ..util import import_module_solver_from_key
from .util import (
    modif_params2d,
    modif_params3d,
    init_parser_base,
    parse_args_dim,
    bench as run_bench,
    tear_down,
    ConsoleError,
)


path_results = "/tmp/fluidsim_bench"
old_print = print
print = mpi.printby0
rank = mpi.rank
nb_proc = mpi.nb_proc
description = "Run benchmarks of FluidSim solvers"


def bench(
    solver,
    dim="2d",
    n0=1024 * 2,
    n1=None,
    n2=None,
    path_dir=None,
    type_fft=None,
    it_end=None,
):
    """Instantiate simulation object and run benchmarks."""

    def _bench(type_fft):
        Simul = solver.Simul
        params = Simul.create_default_params()

        if dim == "2d":
            modif_params2d(params, n0, n1, name_run="bench", type_fft=type_fft)
        elif dim == "3d":
            modif_params3d(
                params, n0, n1, n2, name_run="bench", type_fft=type_fft
            )

        if it_end is not None:
            params.time_stepping.it_end = it_end

        with stdout_redirected():
            sim = Simul(params)

        try:
            run_bench(sim, path_dir)
        except Exception as e:
            if _is_testing:
                raise

            else:
                print(
                    "WARNING: Some error occured while running benchmark"
                    " / saving results!"
                )
                print(e)
        finally:
            tear_down(sim)
            gc.collect()

    if str(type_fft).lower() == "all":
        d = get_opfft(n0, n1, n2, dim, only_dict=True)
        for type_fft, cls in d.items():
            if cls is not None:
                print(type_fft)
                _bench(type_fft)
    else:
        _bench(type_fft)


def get_opfft(n0, n1, n2=None, dim=None, type_fft=None, only_dict=False):
    """Instantiate FFT operator provided by fluidfft."""

    if n2 is None or dim == "2d":
        from fluidfft.fft2d import get_classes_mpi
    elif isinstance(n2, int) or dim == "3d":
        from fluidfft.fft3d import get_classes_mpi

    if mpi.rank == 0:
        d = get_classes_mpi()
    else:
        d = {}

    if mpi.nb_proc > 1:
        d = mpi.comm.bcast(d)

    if only_dict:
        return d

    else:
        if type_fft not in d:
            raise ConsoleError("{} not in {}".format(type_fft, list(d.keys())))

        ClassFFT = d[type_fft]
        if ClassFFT is None:
            raise RuntimeError(f"Class {type_fft} is not available")

        if n2 is None:
            opfft = ClassFFT(n0, n1)
        else:
            opfft = ClassFFT(n0, n1, n2)

        return opfft


def estimate_shapes_weak_scaling(
    n0_max,
    n1_max,
    n2_max=None,
    nproc_min=2,
    nproc_max=None,
    type_fft=None,
    show=False,
):
    """Use this function to get a recommendation of shapeX_seq to initialize the
    solver with to perform weak scaling analysis. The objective is to obtain
    shapeK_loc.

    """
    if nproc_max is None:
        try:
            from os import cpu_count
        except ImportError:
            from multiprocessing import cpu_count

        nproc_max = cpu_count()

    if nproc_min >= nproc_max:
        raise ValueError(
            "Cannot run estimate_shapes_weak_scaling because "
            f"nproc_min >= nproc_max (nproc_max = {nproc_max})"
        )

    if nproc_max % nproc_min != 0:
        raise ValueError(
            "nproc_max % nproc_min != 0 "
            f"(nproc_max={nproc_max}; nproc_min={nproc_min})"
        )

    # Generate a geometric progression for number of processes

    def log(x):
        return int(np.log(x) / np.log(nproc_min))

    num_gp = int(log(nproc_max))
    nproc_gp = np.logspace(1, num_gp, num_gp, base=nproc_min, dtype=int)
    nproc_max = nproc_gp[-1]

    try:
        opfft = get_opfft(n0_max, n1_max, n2_max, type_fft=type_fft)
    except RuntimeError:
        print(f"Cannot create FFT operator {type_fft}")
        return

    shapeX_seq = opfft.get_shapeX_seq()
    shapes = OrderedDict()
    for nproc in nproc_gp:
        divisor = nproc_max // nproc
        if n2_max is None:
            shapes[str(nproc)] = "{} {}".format(
                shapeX_seq[0], shapeX_seq[1] // divisor
            )
        else:
            shapes[str(nproc)] = "{} {}".format(
                shapeX_seq[0], shapeX_seq[1], shapeX_seq[2] // divisor
            )

    if show:
        info._print_heading(["nproc", "shapeX_seq"], case="lower")
        info._print_dict(shapes)

    return shapes


def print_shape_loc(n0, n1, n2=None, type_fft=None):
    """Display the local shape of arrays, shapeX_loc and shapeK_loc.
    Meant to be used with MPI.

    """

    try:
        opfft = get_opfft(n0, n1, n2, type_fft=type_fft)
    except RuntimeError:
        print(f"Cannot create FFT operator {type_fft}")
        return

    shapeX_loc = opfft.get_shapeX_loc()
    shapeK_loc = opfft.get_shapeK_loc()
    print("-" * 8)
    print("type fft = ", type_fft)
    old_print(
        "rank {}: shapeX_loc = {}, shapeK_loc = {}".format(
            rank, shapeX_loc, shapeK_loc
        )
    )


def init_parser(parser):
    """Initialize argument parser for `fluidsim bench`."""

    init_parser_base(parser)
    parser.add_argument("-o", "--output_dir", default=path_results)
    parser.add_argument(
        "-t",
        "--type-fft",
        default=None,
        help=(
            'specify FFT type key (for eg. "fft2d.mpi_with_fftw1d") or "all";'
            "if not specified uses the default FFT method in operators"
        ),
    )
    parser.add_argument(
        "-l",
        "--list-type-fft",
        action="store_true",
        help="list FFT types available for the specified shape or dimension",
    )
    parser.add_argument(
        "-p",
        "--print-shape-loc",
        action="store_true",
        help="mpirun with this option to see how the FFT is initialized",
    )
    parser.add_argument(
        "-e",
        "--estimate-shapes",
        action="store_true",
        help="estimate shapes to plan weak scaling benchmarks",
    )

    parser.add_argument(
        "-it", "--it-end", default=None, type=int, help="Number of iterations"
    )


def run(args):
    """Run `fluidsim bench` command."""

    args = parse_args_dim(args)

    if args.list_type_fft:
        print("FFT classes available for", args.dim.upper())
        d = get_opfft(args.n0, args.n1, args.n2, dim=args.dim, only_dict=True)
        info._print_dict(d)
    elif args.print_shape_loc:
        if args.type_fft is None:
            raise ValueError(
                "Add the fft type, for example -t fft2d.mpi_with_fftw1d"
            )
        print_shape_loc(args.n0, args.n1, args.n2, args.type_fft)
    elif args.estimate_shapes:
        if args.type_fft is None:
            raise ValueError(
                "Add the fft type, for example -t fft2d.mpi_with_fftw1d"
            )
        try:
            estimate_shapes_weak_scaling(
                args.n0, args.n1, args.n2, type_fft=args.type_fft, show=True
            )
        except ValueError as error:
            print(error)

    else:
        # Initialize simulation and run benchmarks
        solver = import_module_solver_from_key(args.solver)
        if args.dim == "3d":
            bench(
                solver,
                args.dim,
                args.n0,
                args.n1,
                args.n2,
                path_dir=args.output_dir,
                type_fft=args.type_fft,
                it_end=args.it_end,
            )
        else:
            bench(
                solver,
                args.dim,
                args.n0,
                args.n1,
                path_dir=args.output_dir,
                type_fft=args.type_fft,
                it_end=args.it_end,
            )
