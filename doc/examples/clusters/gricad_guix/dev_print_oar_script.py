from fluiddyn.clusters.gricad import DahuGuixDevel as Cluster

cluster = Cluster(
    options_guix_shell="-E ^OMPI -E ^OAR -E ^OMP -m manifest.scm -f python-fluidsim.scm"
)
# 2 full nodes
nb_nodes = 2
nb_mpi_processes = "auto"

# 2 cores on 1 node
nb_nodes = 1
nb_mpi_processes = 2

nb_cores_per_node = None

nb_cores_per_node, nb_mpi_processes = cluster._parse_cores_procs(
    nb_nodes, nb_cores_per_node, nb_mpi_processes
)

txt = cluster._create_txt_launching_script(
    command="--prefix $MPI_PREFIX \\\n       fluidsim-bench 128 -d 3 -s ns3d -o .",
    name_run="bench_fluidsim",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node,
    walltime="00:30:00",
    nb_mpi_processes=nb_mpi_processes,
    project="pr-strat-turb",
)
print(txt)
