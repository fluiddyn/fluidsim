from fluidazzurra import cluster

N = 3
Rb = 3

nz = 40
max_elapsed = "00:15:00"
type_fft = "default"  # "fluidfft.fft3d.mpi_with_pfft"

nb_nodes = 1
nb_cores_per_node = cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node


cluster.submit_script(
    (
        f"run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end 5 "
        f"--max-elapsed {max_elapsed} "
    ),
    account="turbulence",
    name_run=f"ns3d.strat_N{N}_Rb{Rb}",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node,
    nb_mpi_processes=nb_mpi_processes,
    omp_num_threads=1,
    ask=True,
    walltime="00:20:00",
)
