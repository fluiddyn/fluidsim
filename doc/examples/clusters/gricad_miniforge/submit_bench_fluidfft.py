#!/usr/bin/env python3
from fluiddyn.clusters.gricad import Dahu32_6130 as Cluster

cluster = Cluster()

for nb_nodes in [1]:
    cluster.submit_command(
        command="fluidfft-bench 1024 -d 3",
        name_run=f"bench_fluidfft_{nb_nodes}nodes",
        nb_nodes=nb_nodes,
        nb_mpi_processes="auto",
        walltime="01:00:00",
        project="pr-strat-turb",
    )
