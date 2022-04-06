"""
submit_tests.py
===============

"""

from fluidlicallo import cluster

nb_proc = nb_cores = 2
walltime = "00:10:00"

cluster.submit_command(
    "fluidsim-test -v",
    name_run="tests_fluidsim",
    nb_cores_per_node=nb_cores,
    walltime=walltime,
    nb_mpi_processes=nb_proc,
    omp_num_threads=1,
    ask=True,
)
