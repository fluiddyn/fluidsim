"""Run benchmarks (:mod:`fluidsim.util.console.bench`)
======================================================

"""
from __future__ import print_function, division

import numpy as np

from fluiddyn.util import mpi, info
from fluiddyn.io import stdout_redirected

from ..util import import_module_solver_from_key
from .util import (
    modif_params2d, modif_params3d, init_parser_base,
    parse_args_dim, bench as run_bench, tear_down)


path_results = '/tmp/fluidsim_bench'
old_print = print
print = mpi.printby0
rank = mpi.rank
nb_proc = mpi.nb_proc


def bench(
        solver, dim='2d', n0=1024 * 2, n1=None, n2=None, path_dir=None,
        type_fft=None):
    """Instantiate simulation object and run benchmarks."""

    Simul = solver.Simul
    params = Simul.create_default_params()

    if dim == '2d':
        modif_params2d(params, n0, n1, name_run='bench', type_fft=type_fft)
    elif dim == '3d':
        modif_params3d(params, n0, n1, n2, name_run='bench', type_fft=type_fft)
    else:
        raise ValueError("dim has to be in ['2d', '3d']")

    try:
        with stdout_redirected():
            sim = Simul(params)
            run_bench(sim, path_dir)
    except Exception:
        print('WARNING: Some error occured while saving results!')
    finally:
        tear_down(sim)


def get_opfft(n0, n1, n2=None, dim=None, type_fft=None, show=False):
    """Instantiate FFT operator provided by fluidfft."""

    if n2 is None or dim == '2d':
        from fluidfft.fft2d import get_classes_mpi
        d = get_classes_mpi()
    elif isinstance(n2, int) or dim == '3d':
        from fluidfft.fft3d import get_classes_mpi
        d = get_classes_mpi()

    if show:
        info._print_dict(d)
    else:
        if type_fft not in d:
            raise ValueError('{} not in {}'.format(type_fft, list(d.keys())))

        if n2 is None:
            opfft = d[type_fft](n0, n1)
        else:
            opfft = d[type_fft](n0, n1, n2)

        return opfft


def estimate_shapes_weak_scaling(
        n0_max, n1_max, n2_max=None, nproc_min=2, nproc_max=None,
        type_fft=None, show=False):
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

    assert nproc_max % nproc_min == 0

    # Generate a geometric progression of powers
    power_max = int(np.log(nproc_max) / np.log(nproc_min))
    start = nproc_min
    ratio = nproc_min
    nproc_gp = [start * ratio ** power for power in range(power_max)]

    opfft = get_opfft(n0_max, n1_max, n2_max, type_fft=type_fft)
    shapeX_seq = opfft.get_shapeX_seq()
    shapes = dict()
    for nproc in nproc_gp:
        divisor = nproc_max // nproc
        if n2_max is None:
            shapes[str(nproc)] = (
                shapeX_seq[0], shapeX_seq[1] // divisor)
        else:
            shapes[str(nproc)] = (
                shapeX_seq[0], shapeX_seq[1], shapeX_seq[2] // divisor)

    if show:
        info._print_heading(['nproc', 'shapeX_seq'], case='lower')
        info._print_dict(shapes)

    return shapes


def print_shape_loc(n0, n1, n2=None, type_fft=None):
    """Display the local shape of arrays, shapeX_loc and shapeK_loc.
    Meant to be used with MPI.

    """
    opfft = get_opfft(n0, n1, n2, type_fft=type_fft)
    shapeX_loc = opfft.get_shapeX_loc()
    shapeK_loc = opfft.get_shapeK_loc()
    print('-' * 8)
    print('type fft = ', type_fft)
    print('shapeX_loc = {}, shapeK_loc = {}'.format(
        shapeX_loc, shapeK_loc))


def init_parser(parser):
    """Initialize argument parser for `fluidsim bench`."""

    init_parser_base(parser)
    parser.add_argument('-o', '--output_dir', default=path_results)
    parser.add_argument('-t', '--type-fft', default=None)
    parser.add_argument(
        '-l', '--list-type-fft', action='store_true',
        help='list FFT types available for the specified shape or dimension')
    parser.add_argument(
        '-p', '--print-shape-loc', action='store_true',
        help='mpirun with this option to see how the FFT is initialized')
    parser.add_argument(
        '-e', '--estimate-shapes', action='store_true',
        help='estimate shapes to plan weak scaling benchmarks')


def run(args):
    """Run `fluidsim bench` command."""

    args = parse_args_dim(args)

    if args.list_type_fft:
        get_opfft(args.n0, args.n1, args.n2, dim=args.dim, show=True)
    elif args.print_shape_loc:
        if args.type_fft is None:
            args.type_fft = 'fft2d.mpi_with_fftw1d'
        print_shape_loc(args.n0, args.n1, args.n2, args.type_fft)
    elif args.estimate_shapes:
        if args.type_fft is None:
            args.type_fft = 'fft2d.mpi_with_fftw1d'
        estimate_shapes_weak_scaling(
            args.n0, args.n1, args.n2, type_fft=args.type_fft, show=True)
    else:
        # Initialize simulation and run benchmarks
        solver = import_module_solver_from_key(args.solver)
        if args.dim == '3d':
            bench(
                solver, args.dim, args.n0, args.n1, args.n2,
                path_dir=args.output_dir, type_fft=args.type_fft)
        else:
            bench(
                solver, args.dim, args.n0, args.n1, path_dir=args.output_dir,
                type_fft=args.type_fft)
