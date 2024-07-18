from ciment import DahuGuix16_6130 as Cluster

cluster = Cluster()


cluster.submit_command(
    command="--prefix $MPI_PREFIX fluidsim-bench 128 -d 3 -s ns3d -o .",
    name_run="bench_fluidsim",
    nb_nodes=2,
    nb_mpi_processes="auto",
    walltime="00:30:00",
    project="pr-strat-turb",
    submit=False,
    devel=True,
)
