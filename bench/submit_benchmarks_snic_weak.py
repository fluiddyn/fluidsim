#!/usr/bin/env python
import numpy as np
from submit_benchmarks_snic import (
    init_cluster, submit, nb_cores, nodes, n0, mode, argv)


argv['weak'] = True

divisors = np.array(nb_cores[::-1]) // 2
multipliers = np.array(nodes)
n1_intra = (n0 / divisors).astype(np.int)
n1_inter = n0 * multipliers


if __name__ == '__main__':
    cluster, interactive = init_cluster()
    if 'intra' in mode:
        nb_nodes = 1
        for type_fft in argv['fft']:
            for nb_cores_per_node, n1 in zip(nb_cores, n1_intra):
                if nb_cores_per_node > cluster.nb_cores_per_node:
                    continue
                shape = '{} {}'.format(n0, n1)
                submit(
                    cluster, interactive, nb_nodes, nb_cores_per_node,
                    shape=shape, fft=type_fft)

    if 'inter' in mode:
        for type_fft in argv['fft']:
            for nb_nodes, n1 in zip(nodes, n1_inter):
                shape = '{} {}'.format(n0, n1)
                submit(cluster, interactive, nb_nodes, shape=shape,
                       fft=type_fft)

