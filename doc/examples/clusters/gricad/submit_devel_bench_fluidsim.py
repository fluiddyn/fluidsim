from fluiddyn.clusters.gricad import DahuGuixDevel as Cluster

cluster = Cluster(
    check_scheduler=False,
    options_guix_shell="-E ^OMPI -E ^OAR -E ^OMP -m manifest.scm -f python-fluidsim.scm",
)

cluster.submit_command(
    command="--prefix $MPI_PREFIX \\\n       fluidsim-bench 256 -d 3 -s ns3d -o .",
    name_run="bench_fluidsim",
    nb_nodes=1,
    nb_mpi_processes=2,
    walltime="00:30:00",
    project="pr-strat-turb",
)
