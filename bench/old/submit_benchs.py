from fluiddyn.clusters.legi import Calcul8 as Cluster

cluster = Cluster()

# cluster.submit_script(
#     'simul_bench.py', name_run='fluidsim_bench',
#     omp_num_threads=1, walltime='1:00:00')


for nb_mpi_processes in [2, 4, 8, 16]:
    cluster.submit_script(
        "simul_bench.py",
        name_run="fluidsim_bench",
        walltime="1:00:00",
        omp_num_threads=1,
        nb_cores_per_node=nb_mpi_processes,
        nb_mpi_processes=nb_mpi_processes,
        ask=False,
    )
