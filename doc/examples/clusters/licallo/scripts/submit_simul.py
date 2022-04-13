from fluidlicallo import cluster

# Ns = [10, 20, 40]
# Rbs = [10, 20, 40]
# projs = [None, "poloidal"]

Ns = [3]
Rbs = [3]
projs = ['"poloidal"']

nz = 80
max_elapsed = "00:15:00"
type_fft = "'fluidfft.fft3d.mpi_with_pfft'"

nb_nodes = 1
nb_cores_per_node = 8  # cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node

for N in Ns:
    for Rb in Rbs:
        for proj in projs:
            cluster.submit_command(
                (
                    f"./run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end 20 "
                    f"--projection={proj} --spatiotemporal-spectra "
                    f"--max-elapsed {max_elapsed} "
                    f'--modify-params "params.oper.type_fft = {type_fft};"'
                ),
                name_run=f"ns3d.strat_N{N}_Rb{Rb}_proj{proj}",
                nb_nodes=nb_nodes,
                nb_cores_per_node=nb_cores_per_node,
                nb_mpi_processes=nb_mpi_processes,
                omp_num_threads=1,
                ask=True,
                walltime="00:20:00",
            )
