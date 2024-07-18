from ciment import DahuGuix16_6130 as Cluster

cluster = Cluster()

nb_nodes = 2
nb_cores_per_node = None
nb_mpi_processes = "auto"

nb_cores_per_node, nb_mpi_processes = cluster._parse_cores_procs(
    nb_nodes, nb_cores_per_node, nb_mpi_processes
)

txt = cluster._create_txt_launching_script(
    command="--prefix $MPI_PREFIX fluidsim-bench 128 -d 3 -s ns3d -o .",
    name_run="bench_fluidsim",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node,
    walltime="00:30:00",
    nb_mpi_processes=nb_mpi_processes,
    devel=True,
    project="pr-strat-turb",
)
print(txt)
