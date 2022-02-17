from fluidlicallo import cluster

# Ns = [10, 20, 40]
# Rbs = [10, 20, 40]
# projs = ["None", "poloidal"]

Ns = [3]
Rbs = [300000000]
projs = ["None"]
nz = 20

nb_nodes = 1
nb_cores_per_node = 2 # cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = 2 # nb_nodes * nb_cores_per_node

walltime = "00:30:00"
max_elapsed = "00:25:00"
# type_fft = "default" #"'fluidfft.fft3d.mpi_with_fftw1d'"


for N in Ns:
    for Rb in Rbs:
        for proj in projs:
            if proj == "None":
                command = f'run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end 10 --max-elapsed {max_elapsed}'
            else:
                command = f'run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end 10 --projection={proj} --max-elapsed {max_elapsed}'
            cluster.submit_script(
                f"{command}",
                name_run=f"ns3d.strat_N{N}_Rb{Rb}",
                nb_nodes=nb_nodes,
                nb_cores_per_node=nb_cores_per_node,
                nb_mpi_processes=nb_mpi_processes,
                omp_num_threads=1,
                ask=True,
                walltime=walltime,
            )
