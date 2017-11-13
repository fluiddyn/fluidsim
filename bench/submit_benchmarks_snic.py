#!/usr/bin/env python
import os
from fluiddyn.clusters.snic import ClusterSNIC as Cluster
import fluidsim
from fluidsim.util.console import bench


# Parameters
# ----------
# n0 = 2 ** 10; 'Triolith / Beskow'
n0 = 1024; nb_cores = [2, 4, 8, 16, 32]; nodes = [2, 4, 8]
# n0 = 2**6 * 3**2 * 7; 'Kebnekaise'
# n0 = 1008; nb_cores = [2, 4, 8, 12, 16, 21, 24, 28]; nodes = [2, 3, 4, 6]

# 2D benchmarks
argv = dict(
    dim='2d', nh='{} {}'.format(n0, n0), time='00:20:00',
    weak=True,
    fft=[
        'fft2d.mpi_with_fftw1d',
        'fft2d.mpi_with_fftwmpi2d',
    ]

# 3D benchmarks
# argv = dict(dim='3d', nh='960 960 240', time='00:50:00')

solver = 'ns2d'

# mode = 'intra'
# mode = 'inter'
mode = 'inter-intra'

dry_run = False
# dry_run = True


def init_cluster():
    global output_dir

    cluster = Cluster()
    if cluster.name_cluster == 'beskow':
        cluster.default_project = '2016-34-10'
        interactive = True
    else:
        cluster.default_project = 'SNIC2016-34-10'
        interactive = True

    output_dir = os.path.abspath('{}/../doc/benchmarks/snic_{}_{}'.format(
        os.path.split(fluidsim.__file__)[0], cluster.name_cluster, argv['dim']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Output directory: ', output_dir)
    cluster.commands_unsetting_env.insert(0, 'fluidinfo -o ' + output_dir)
    return cluster, interactive


def submit(cluster, interactive, nb_nodes, nb_cores_per_node=None, shape=None, fft='all'):
    if nb_cores_per_node is None:
        nb_cores_per_node = cluster.nb_cores_per_node

    nb_mpi = nb_cores_per_node * nb_nodes
    cmd = 'fluidsim bench -s {} {} -t {} -o {}'.format(solver, shape, fft, output_dir)
    if dry_run:
        print('nb_mpi = ', nb_mpi, end=' ')
        print(cmd)
    else:
        cluster.submit_command(
            cmd,
            name_run='{}_{}'.format(solver, nb_mpi),
            nb_nodes=nb_nodes,
            nb_cores_per_node=nb_cores_per_node,
            walltime=argv['time'],
            nb_mpi_processes=nb_mpi, omp_num_threads=1,
            ask=False, bash=False, interactive=interactive)


if __name__ == '__main__':
    cluster, interactive = init_cluster()
    if argv['weak']:
        _shapes = [int(n) for n in argv['nh'].split()]
        _shapes = bench.estimate_shapes_weak_scaling(
            *_shapes, nproc_max=nodes[-1] * cluster.nb_cores_per_node)

    if 'intra' in mode:
        nb_nodes = 1
        for nb_cores_per_node, type_fft in zip(nb_cores, argv['fft']):
            if nb_cores_per_node > cluster.nb_cores_per_node:
                continue
            if argv['weak']:
                try:
                    shape = _shapes[nb_cores_per_node]
                except KeyError:
                    continue
            else:
                shape = argv['nh']

            submit(cluster, interactive, nb_nodes, nb_cores_per_node, shape=shape, fft=type_fft)

    if 'inter' in mode:
        for nb_nodes, type_fft in zip(nodes, argv['fft']):
            if argv['weak']:
                try:
                    shape = _shapes[nb_nodes * cluster.nb_cores_per_node]
                except KeyError:
                    continue
            else:
                shape = argv['nh']

            submit(cluster, interactive, nb_nodes, shape=shape, fft=type_fft)
