from fluiddyn.clusters.gricad import DahuGuix16_6130 as Cluster

cluster = Cluster(
    check_scheduler=False,
    options_guix_shell="-E ^OMPI -E ^OAR -E ^OMP -m manifest.scm -f python-fluidsim.scm",
)

for nb_nodes in [1, 2, 4]:
    cluster.submit_command(
        command="--prefix $MPI_PREFIX \\\n       fluidfft-bench 1024 -d 3",
        name_run=f"bench_fluidfft_{nb_nodes}nodes",
        nb_nodes=nb_nodes,
        nb_mpi_processes="auto",
        walltime="01:00:00",
        project="pr-strat-turb",
    )
