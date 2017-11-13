"""Utilities for benchmarking and profiling (:mod:`fluidsim.util.bench.util`)
=============================================================================

"""
from __future__ import division
import os
from time import time
try:
    from time import perf_counter as clock
except ImportError:
    # python 2.7
    from time import clock
import pstats
import cProfile
import json
import socket
import shutil

from fluiddyn.util import time_as_str
from fluiddyn.util import mpi
from ..util import available_solver_keys


class MyValueError(ValueError):
    """Bypass errors."""
    pass


def modif_box_size(params, n0, n1, n2=None):
    """Modify box size, such that the aspect ration is square

    Parameters
    ----------
    params : ParamContainer
        Input parameters
    n0 : int
        Number of grid points in x-axis of the grid
    n1 : int
        Number of grid points in y-axis of the grid
    n2 :
        Number of grid points in z-axis of the grid

    Returns
    -------

    """
    if n1 != n0:
        if n1 < n0:
            params.oper.Ly = params.oper.Ly * n1 / n0
        else:
            params.oper.Lx = params.oper.Lx * n0 / n1

    if n2 is not None and n2 != n0:
        if n2 < n0:
            params.oper.Lz = params.oper.Lz * n2 / n0
        else:
            params.oper.Lx = params.oper.Lx * n0 / n2
            # Recurse to correct x-y aspect ratio
            modif_box_size(params, n0, n1, n2)


def modif_params2d(
        params, n0=3 * 2**8, n1=None, name_run='profile', type_fft=None):
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

    params.oper.nx = n0
    params.oper.ny = n1
    modif_box_size(params, n0, n1)

    if type_fft is not None:
        params.oper.type_fft = type_fft

    if 'FLUIDSIM_NO_FLUIDFFT' not in os.environ:
        # params.oper.type_fft = 'fft2d.mpi_with_fftwmpi2d'
        pass

    params.FORCING = True
    params.forcing.type = 'tcrandom'
    params.forcing.nkmax_forcing = 5
    params.forcing.nkmin_forcing = 4
    params.forcing.forcing_rate = 1.

    params.nu_8 = 1.
    try:
        params.f = 1.
        params.c2 = 200.
    except AttributeError:
        pass

    try:
        params.N = 1.
    except AttributeError:
        pass

    params.time_stepping.deltat0 = 1.e-4
    params.time_stepping.USE_CFL = False
    params.time_stepping.it_end = 1000
    params.time_stepping.USE_T_END = False

    params.output.periods_print.print_stdout = 0
    params.output.HAS_TO_SAVE = 0


def modif_params3d(
        params, n0=256, n1=None, n2=None, name_run='profile', type_fft=None):
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

    params.oper.nx = n0
    params.oper.ny = n1
    params.oper.nz = n2
    modif_box_size(params, n0, n1, n2)

    if type_fft is not None:
        params.oper.type_fft = type_fft

    if 'FLUIDSIM_NO_FLUIDFFT' not in os.environ:
        # params.oper.type_fft = 'fft2d.mpi_with_fftwmpi2d'
        pass

    # params.FORCING = False
    # params.forcing.type = 'tcrandom'
    # params.forcing.nkmax_forcing = 5
    # params.forcing.nkmin_forcing = 4
    # params.forcing.forcing_rate = 1.

    params.nu_8 = 1.
    try:
        params.f = 1.
        params.c2 = 200.
    except AttributeError:
        pass

    try:
        params.N = 1.
    except AttributeError:
        pass

    params.time_stepping.deltat0 = 1.e-4
    params.time_stepping.USE_CFL = False
    params.time_stepping.it_end = 100
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
    parser.add_argument('n0', nargs='?', type=int, default=64)
    parser.add_argument('n1', nargs='?', type=int, default=None)
    parser.add_argument('n2', nargs='?', type=int, default=None)

    parser.add_argument(
        '-s', '--solver', type=str, default='ns2d',
        help='Any of the following solver keys: {}'.format(
            available_solver_keys())
    )
    parser.add_argument('-d', '--dim', default='2')


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

    if dim is None:
        if n0 is not None and n1 is not None and n2 is None:
            dim = '2d'
        elif n0 is not None and n1 is not None and n2 is not None:
            dim = '3d'
        else:
            print(
                'Cannot determine which shape you want to use for this bench '
                "('2d' or '3d')")
            raise MyValueError

    if dim.lower() in ['3', '3d']:
        if n2 is None:
            n2 = n0
        dim = '3d'
    elif dim.lower() in ['2', '2d']:
        dim = '2d'
    else:
        raise ValueError('dim should not be {}'.format(dim))

    if n1 is None:
        n1 = n0

    args.dim = dim
    args.n0 = n0
    args.n1 = n1
    args.n2 = n2

    return args


def profile(sim, nb_dim=2):
    """Profile a simulation run and save the results in `profile.pstats`

    Parameters
    ----------
    sim : Simul
        An initialized simulation object
    nb_dim : int
        Dimension of the solver

    """
    t0 = time()

    cProfile.runctx('sim.time_stepping.start()',
                    globals(), locals(), 'profile.pstats')
    t_end = time()
    if sim.oper.rank == 0:
        s = pstats.Stats('profile.pstats')
        # s.strip_dirs().sort_stats('time').print_stats(16)
        s.sort_stats('time').print_stats(12)

        if nb_dim == 2:
            times = print_analysis(s)
        elif nb_dim == 3:
            times = print_analysis3d(s)
        else:
            raise NotImplementedError

        print('\nelapsed time = {:.3f} s'.format(t_end - t0))

        print(
            '\nwith gprof2dot and graphviz (command dot):\n'
            'gprof2dot -f pstats profile.pstats | dot -Tpng -o profile.png')


def bench(sim, path_results):
    """Benchmark a simulation run and save the results in a JSON file.

    Parameters
    ----------
    sim : Simul
        An initialized simulation object
    path_results :  str
        Directory path to save results in

    """
    t_as_str = time_as_str()
    t0_usr = time()
    t0_sys = clock()
    sim.time_stepping.start()
    t_elapsed_sys = clock() - t0_sys
    t_elapsed_usr = time() - t0_usr

    if sim.oper.rank != 0:
        return

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    key_solver = sim.info_solver.short_name.lower()
    pid = os.getpid()
    nfile = (
        'result_bench_' + key_solver + '_' +
        sim.oper.produce_str_describing_grid() +
        '_' + t_as_str + '_{}'.format(pid) + '.json')

    path = os.path.join(path_results, nfile)

    results = {
        't_elapsed_usr': t_elapsed_usr,
        't_elapsed_sys': t_elapsed_sys,
        'key_solver': sim.info_solver.short_name.lower(),
        'n0': sim.oper.shapeX_seq[0],
        'n1': sim.oper.shapeX_seq[1],
        'n0_loc': sim.oper.shapeX_loc[0],
        'n1_loc': sim.oper.shapeX_loc[1],
        'k0_loc': sim.oper.shapeK_loc[0],
        'k1_loc': sim.oper.shapeK_loc[1],
        'nb_proc': mpi.nb_proc,
        'pid': pid,
        'time_as_str': t_as_str,
        'hostname': socket.gethostname(),
        'nb_iter': sim.params.time_stepping.it_end,
        'type_fft': sim.oper.type_fft,
    }

    try:
        results.update({
            'n2': sim.oper.shapeX_seq[2],
            'n2_loc': sim.oper.shapeX_loc[2],
            'k2_loc': sim.oper.shapeK_loc[2]
        })
    except IndexError:
        pass

    with open(path, 'w') as f:
        json.dump(results, f, sort_keys=True)
        f.write('\n')

    print('\nresults benchmarks saved in\n' + path + '\n')


def tear_down(sim):
    """Delete simulation directory."""
    if mpi.rank == 0:
        print('Cleaning up simulation.')
        shutil.rmtree(sim.output.path_run)


def print_analysis(s):
    """Print analysis of profiling result of a 2D solver.

    Parameters
    ----------
    s : pstats.Stats
        Object pointing to a stats file

    """
    total_time = 0.
    times = {'fft2d': 0., 'fft_as': 0., 'pythran': 0., '.pyx': 0.}
    for key, value in s.stats.items():
        name = key[2]
        time = value[2]
        total_time += time
        for k in times.keys():
            if k in name or k in key[0]:
                if k == '.pyx':
                    if 'fft/Sources' in key[0]:
                        continue
                    if 'fft_as_arg' in key[2]:
                        continue

                if k == 'fft2d':

                    if 'util_pythran' in key[2] or \
                       'operators.py' in key[0] or \
                       'fft_as_arg' in key[2]:
                        continue

                    callers = value[4]

                    time = 0
                    for kcaller, vcaller in callers.items():
                        if 'fft_as_arg' not in kcaller[2] and\
                           'fft_as_arg' not in kcaller[0]:
                            time += vcaller[2]

                    # print(k, key)
                    # print(value[:100])
                    # print(time, '\n')

                if k == 'fft_as':
                    if '.pyx' in key[0]:
                        continue
                    # time = value[3]

                    # print(k, key)
                    # print(value[:100])
                    # print(time, '\n')

                times[k] += time

    print('Analysis (percentage of total time):')

    keys = list(times.keys())
    keys.sort(key=lambda key: times[key], reverse=True)

    for k in keys:
        t = times[k]
        print('time {:10s}: {:5.01f} % ({:4.02f} s)'.format(
            k, t / total_time * 100, t))

    print('-' * 24 + '\n{:15s}  {:5.01f} %'.format(
        '', sum([t for t in times.values()]) / total_time * 100))

    return times


def print_analysis3d(s):
    """Print analysis of profiling result of a 3D solver.

    Parameters
    ----------
    s : pstats.Stats
        Object pointing to a stats file

    """
    total_time = 0.
    times = {'fft3d': 0., 'fft_as': 0., 'pythran': 0., '.pyx': 0.}
    for key, value in s.stats.items():
        name = key[2]
        time = value[2]
        total_time += time
        for k in times.keys():
            if k in name or k in key[0]:

                if k == 'fft3d':
                    if 'pythran' in key[0]:
                        continue
                    if 'operators.py' in key[0]:
                        continue

                # print(k, key)
                # print(value[:100])
                # print(time, '\n')

                times[k] += time

    print('Analysis (percentage of total time):')

    keys = list(times.keys())
    keys.sort(key=lambda key: times[key], reverse=True)

    for k in keys:
        t = times[k]
        print('time {:10s}: {:5.01f} % ({:4.02f} s)'.format(
            k, t / total_time * 100, t))

    print('-' * 24 + '\n{:15s}  {:5.01f} %'.format(
        '', sum([t for t in times.values()]) / total_time * 100))

    return times
