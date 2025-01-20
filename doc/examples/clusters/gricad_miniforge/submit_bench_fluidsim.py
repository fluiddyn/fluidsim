#!/usr/bin/env python3
from fluiddyn.clusters.gricad import Dahu32_6130 as Cluster

cluster = Cluster()

cluster.submit_command(
    command="fluidsim-bench 1024 -d 3 -s ns3d -o .",
    name_run="bench_fluidsim",
    nb_nodes=1,
    nb_mpi_processes="auto",
    walltime="00:30:00",
    project="pr-strat-turb",
)
