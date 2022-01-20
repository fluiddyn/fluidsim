import sys

from fluiddyn.clusters.idris import JeanZay as Cluster

cluster = Cluster()

nb_nodes = 1
nb_cores_per_node = cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node

walltime = "00:40:00"

cluster.commands_setting_env += [
    "conda activate env_fluidsim",  # TODO: maybe this line should go into fluiddyn
    "export FLUIDSIM_PATH=$WORK/Fluidsim_Data",
]

# TODO: We could do a more usefull example with several runs like for occigen

# TODO: Check number of mpi processes (like assert not ny % nb_mpi_processes)

cluster.submit_script(
    "run_simul.py",
    name_run=f"ns3d.strat",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node, # it is computed automatically I think
    nb_mpi_processes=nb_mpi_processes,
    omp_num_threads=1,
    ask=True,
    walltime=walltime,
)


