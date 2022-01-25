"""
submit_tests.py
===============

"""

from fluidjean_zay import cluster

nb_proc = nb_cores = 2
walltime = "00:10:00"

cluster.commands_setting_env += [
    "export FLUIDSIM_PATH=$WORK/Fluidsim_Data/tests_fluidsim",
]

cluster.submit_command(
    "fluidsim-test -v",
    name_run="tests_fluidsim",
    nb_cores_per_node=nb_cores,
    walltime=walltime,
    nb_mpi_processes=nb_proc,
    omp_num_threads=1,
    ask=True,
)
