from ciment import DahuGuixDevel as Cluster

cluster = Cluster()

cluster.submit_command(
    command="--prefix $MPI_PREFIX \\\n       fluidsim-bench 256 -d 3 -s ns3d -o .",
    name_run="bench_fluidsim",
    nb_nodes=1,
    nb_mpi_processes=2,
    walltime="00:30:00",
    project="pr-strat-turb",
)
