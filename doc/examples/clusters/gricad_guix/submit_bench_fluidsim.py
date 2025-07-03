#!/usr/bin/env python

from dahu import Dahu32_6130 as Cluster

cluster = Cluster(
    check_scheduler=False,
)

cluster.submit_command(
    command="fluidsim-bench 1024 -d 3 -s ns3d -o .",
    name_run="bench_fluidsim",
    nb_nodes=2,
    nb_mpi_processes="auto",
    walltime="00:30:00",
    project="pr-strat-turb",
)
