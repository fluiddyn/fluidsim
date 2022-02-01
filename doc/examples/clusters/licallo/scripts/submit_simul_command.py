from fluidlicallo import cluster

nb_nodes = 1
nb_cores_per_node = 8  # cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node

walltime = "00:20:00"

cluster.submit_command(
    f"run_simul.py -N 10 -nz 80",
    name_run=f"ns3d.strat_test",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node,
    nb_mpi_processes=nb_mpi_processes,
    omp_num_threads=1,
    ask=True,
    walltime=walltime,
)
