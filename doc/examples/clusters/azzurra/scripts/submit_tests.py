"""
submit_tests.py
===============

"""

from fluidazzurra import cluster
import mpi4py
mpi4py.rc(thread_level = 'single')

nb_proc = nb_cores = 2
walltime = "00:10:00"

cluster.submit_command(
    "fluidsim-test -v",
    account="turbulence",
    name_run="tests_fluidsim",
    nb_cores_per_node=nb_cores,
    walltime=walltime,
    nb_mpi_processes=nb_proc,
    omp_num_threads=1,
    ask=True,
)
