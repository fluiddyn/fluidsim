"""
submit_tests.py
===============

"""

from fluiddyn.clusters.idris import JeanZay as Cluster

cluster = Cluster()

nb_proc = nb_cores = 2
walltime = "00:10:00"

cluster.commands_setting_env += [
    "conda activate env_fluidsim",  # TODO: maybe this line should go into fluiddyn
    "export FLUIDSIM_PATH=$WORK/Fluidsim_Data/tests_fluidfft",
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
