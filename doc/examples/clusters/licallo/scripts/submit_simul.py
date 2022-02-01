from fluidlicallo import cluster

nb_nodes = 1
nb_cores_per_node = 4  # cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node

walltime = "00:10:00"

cluster.submit_script(
    "run_simul.py",
    name_run=f"ns3d.strat",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node,
    nb_mpi_processes=nb_mpi_processes,
    omp_num_threads=1,
    ask=True,
    walltime=walltime,
)
