#!/usr/bin/env python
import os
import numpy as np
from fluiddyn.clusters.snic import ClusterSNIC as Cluster
import fluidfft


# Parameters
# ----------
## n0 = 2 ** 10; 'Triolith / Beskow'
n0 = 1024; nb_cores = [2, 4, 8, 16, 32]; nodes = [2, 4, 8]

## n0 = 2**6 * 3**2 * 7; 'Kebnekaise'
# n0 = 1008; nb_cores = [2, 4, 8, 12, 16, 21, 24, 28]; nodes = [2, 3, 4, 6]

argv = dict(dim='2d', nh='{} -d 2'.format(n0), time='00:04:00')  # 2D benchmarks
# argv = dict(dim='3d', nh='960 960 240', time='00:20:00')  # 3D benchmarks
# mode = 'intra'
# mode = 'inter'
solver = 'ns2d'
mode = 'inter-intra'


def init_cluster():
    global output_dir

    cluster = Cluster()
    if cluster.name_cluster == 'beskow':
        cluster.default_project = '2016-34-10'
        interactive=False
    else:
        cluster.default_project = 'SNIC2016-34-10'
        interactive=True

    output_dir = os.path.abspath('{}/../doc/benchmarks/snic_{}_{}'.format(
        os.path.split(fluidfft.__file__)[0], cluster.name_cluster, argv['dim']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Output directory: ', output_dir)
    cluster.commands_unsetting_env.insert(0, 'fluidinfo -o ' + output_dir)
    return cluster, interactive


def submit(cluster, interactive, nb_nodes, nb_cores_per_node=None):
    if nb_cores_per_node is None:
        nb_cores_per_node = cluster.nb_cores_per_node

    nb_mpi = nb_cores_per_node * nb_nodes
    cluster.submit_command(
        'fluidsim bench -s ' + solver + argv['nh'] + ' -o ' + output_dir,
        name_run='fft{}_{}'.format(argv['dim'], nb_mpi),
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        walltime=argv['time'],
        nb_mpi_processes=nb_mpi, omp_num_threads=1,
        ask=False, bash=False, interactive=interactive)


cluster, interactive = init_cluster()
if 'intra' in mode:
    nb_nodes = 1
    for nb_cores_per_node in nb_cores:
        if nb_cores_per_node > cluster.nb_cores_per_node:
            continue
        submit(cluster, interactive, nb_nodes, nb_cores_per_node)

if 'inter' in mode:
    for nb_nodes in nodes:
        submit(cluster, interactive, nb_nodes)
