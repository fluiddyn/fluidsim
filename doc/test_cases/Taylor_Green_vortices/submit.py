from fluiddyn.clusters.legi import Calcul8 as Cluster

cluster = Cluster()

cluster.commands_setting_env = [
    "source /etc/profile",
    'export PATH="$HOME/miniconda3/bin:$PATH"'
    "export FLUIDSIM_PATH=/fsnet/project/watu/2020/20MILESTONE",
]

nb_nodes = 1
nb_cores_per_node = 16
nb_mpi_processes = nb_nodes * nb_cores_per_node


cluster.submit_script(
    "run_simul.py",
    name_run="RK2",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node,
    nb_mpi_processes=nb_mpi_processes,
    omp_num_threads=1,
    ask=False,
    walltime="23:59:58",
)

cluster.submit_script(
    "run_simul_phaseshift.py",
    name_run="RK2_phaseshift",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node,
    nb_mpi_processes=nb_mpi_processes,
    omp_num_threads=1,
    ask=False,
    walltime="23:59:58",
)
