import sys

from fluiddyn.clusters.idris import JeanZay as Cluster

cluster = Cluster()

nb_proc = nb_cores = 2
walltime = "00:10:00"

cluster.commands_setting_env += [
    "conda activate env_fluidsim",  # TODO: maybe this line should go into fluiddyn
    "export FLUIDSIM_PATH=$WORK/Fluidsim_Data",
]

# TODO: We could do a more usefull example with several runs like for occigen

cluster.submit_script(
    "run_simul.py",
    name_run=f"ns3d.strat",
    nb_cores_per_node=nb_cores,
    nb_mpi_processes=nb_proc,
    omp_num_threads=1,
    ask=True,
    walltime=walltime,
)


