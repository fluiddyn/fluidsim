"""Run profiles (:mod:`fluidsim.util.console.profile`)
======================================================

"""
from __future__ import print_function, division

import gc

from fluiddyn.util import mpi
from fluiddyn.io import stdout_redirected

from ..util import import_module_solver_from_key
from .util import (
    modif_params2d, modif_params3d, init_parser_base,
    parse_args_dim, profile as run_profile, tear_down)

from .bench import get_opfft


path_results = '/tmp/fluidsim_profile'
old_print = print
print = mpi.printby0
rank = mpi.rank
nb_proc = mpi.nb_proc


def profile(
        solver, dim='2d', n0=1024 * 2, n1=None, n2=None, path_dir=None,
        type_fft=None, raise_error=False, verbose=False):
    """Instantiate simulation object and run profiles."""

    def _profile(type_fft):
        """Run profile once."""
        Simul = solver.Simul
        params = Simul.create_default_params()

        if dim == '2d':
            modif_params2d(params, n0, n1, name_run='bench', type_fft=type_fft)
        elif dim == '3d':
            modif_params3d(params, n0, n1, n2, name_run='bench',
                           type_fft=type_fft)
        else:
            raise ValueError("dim has to be in ['2d', '3d']")

        nb_dim = int(dim[0])

        with stdout_redirected():
            sim = Simul(params)

        try:
            if verbose:
                run_profile(sim, nb_dim, path_dir)
            else:
                with stdout_redirected():
                    run_profile(sim, nb_dim, path_dir)
        except Exception as e:
            if raise_error:
                raise
            else:
                print('WARNING: Some error occurred while running benchmark'
                      ' / saving results!')
                print(e)
        finally:
            tear_down(sim)
            gc.collect()

    if str(type_fft).lower() == 'all':
        d = get_opfft(n0, n1, n2, dim, only_dict=True)
        for type_fft, cls in d.items():
            if cls is not None:
                print(type_fft)
                _profile(type_fft)
    else:
        _profile(type_fft)


def init_parser(parser):
    """Initialize argument parser for `fluidsim profile`."""

    init_parser_base(parser)
    parser.add_argument('-o', '--output_dir', default=path_results)
    parser.add_argument(
        '-t', '--type-fft', default=None,
        help=(
            'specify FFT type key (for eg. "fft2d.mpi_with_fftw1d") or "all";'
            'if not specified uses the default FFT method in operators'))
    parser.add_argument('-v', '--verbose', action='store_true')


def run(args):
    """Run `fluidsim profile` command."""

    args = parse_args_dim(args)

    # Initialize simulation and run benchmarks
    solver = import_module_solver_from_key(args.solver)
    if args.dim == '3d':
        profile(
            solver, args.dim, args.n0, args.n1, args.n2,
            path_dir=args.output_dir, type_fft=args.type_fft,
            verbose=args.verbose)
    else:
        profile(
            solver, args.dim, args.n0, args.n1, path_dir=args.output_dir,
            type_fft=args.type_fft, verbose=args.verbose)
