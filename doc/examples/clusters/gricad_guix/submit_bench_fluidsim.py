from fluiddyn.clusters.gricad import DahuGuix16_6130 as Cluster

cluster = Cluster(
    check_scheduler=False,
    options_guix_shell="-E ^OMPI -E ^OAR -E ^OMP -m manifest.scm -f python-fluidsim.scm",
)

cluster.submit_command(
    command="--prefix $MPI_PREFIX \\\n       fluidsim-bench 1024 -d 3 -s ns3d -o .",
    name_run="bench_fluidsim",
    nb_nodes=2,
    nb_mpi_processes="auto",
    walltime="00:30:00",
    project="pr-strat-turb",
)
