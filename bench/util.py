import os
import argparse
import fluidsim
from fluidsim.base.params import Parameters
from fluidsim.util.util import available_solver_keys
import numpy as np


def create_common_params(n0, n1=None, n2=None):
    params = Parameters('submit')
    params._set_attrib('weak', False)
    params._set_attrib('dry_run', False)
    params._set_attrib('mode', 'intra')
    params._set_attrib('dim', 3)
    params._set_attrib('output_dir', '')

    if n1 is None:
        n1 = n0

    params._set_child('two_d', dict(
        shape='{} {}'.format(n0, n1), time='00:20:00',
        solver='ns2d',
        fft=['fft2d.mpi_with_fftw1d',
             'fft2d.mpi_with_fftwmpi2d'],
        nb_cores=np.logspace(1, 8, 8, base=2, dtype=int), nodes=[1]))

    if n2 is None:
        n2 = n0

    params._set_child('three_d', dict(
        shape='{} {} {}'.format(n0, n1, n2), time='00:30:00',
        solver='ns3d',
        fft=['fft3d.mpi_with_fftw1d',
             'fft3d.mpi_with_fftwmpi3d',
             'fft3d.mpi_with_p3dfft',
             'fft3d.mpi_with_pfft'],
        nb_cores=np.logspace(1, 10, 10, base=2, dtype=int),
        nodes=[1]))
    return params


def get_parser(prog='', description=''):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('n0', nargs='?', type=int, default=None)
    parser.add_argument('n1', nargs='?', type=int, default=None)
    parser.add_argument('n2', nargs='?', type=int, default=None)
    parser.add_argument(
        '-s', '--solver', type=str, default=None,
        help='Any of the following solver keys: {}'.format(
            available_solver_keys()))
    parser.add_argument('-d', '--dim', type=int, default=3)

    parser.add_argument('-n', '--dry-run', action='store_true')
    parser.add_argument('-m', '--mode', default='intra')
    return parser


def parser_to_params(parser):
    args = parser.parse_args()
    if args.dim == 3:
        params = create_common_params(args.n0, args.n1, args.n2)
        params_dim = params.three_d
    else:
        params = create_common_params(args.n0, args.n1)
        params_dim = params.two_d

    params.dim = args.dim
    params.dry_run = args.dry_run
    params.mode = args.mode
    return params, params_dim


def init_cluster(params, Cluster, prefix='snic'):

    cluster = Cluster()
    output_dir = params.output_dir = os.path.abspath(
        './../doc/benchmarks/{}_{}_{}d'.format(
            prefix, cluster.name_cluster, params.dim))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Output directory: ', output_dir)
    cluster.commands_unsetting_env.insert(0, 'fluidinfo -o ' + output_dir)
    return cluster


def submit(
        params, params_dim, cluster, nb_nodes, nb_cores_per_node=None, fft='all'):
    if nb_cores_per_node is None:
        nb_cores_per_node = cluster.nb_cores_per_node

    nb_mpi = nb_cores_per_node * nb_nodes
    cmd = 'fluidsim bench -s {} {} -t {} -o {}'.format(
        params_dim.solver, params_dim.shape, fft, params.output_dir)
    if params.dry_run:
        print('nb_mpi = ', nb_mpi, end=' ')
        print(cmd)
    else:
        cluster.submit_command(
            cmd,
            name_run='{}_{}'.format(params_dim.solver, nb_mpi),
            nb_nodes=nb_nodes,
            nb_cores_per_node=nb_cores_per_node,
            walltime=params_dim.time,
            nb_mpi_processes=nb_mpi, omp_num_threads=1,
            ask=False, bash=False, interactive=True, retain_script=False)
