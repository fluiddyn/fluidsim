from fluidlicallo import cluster

# Ns = [10, 20, 40]
# Rbs = [10, 20, 40]
# projs = ["None", "poloidal"]

Ns = [40]
Rbs = [80]
projs = ["None"]
nz = 80

nb_nodes = 1
nb_cores_per_node = 40 # cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = 40 # nb_nodes * nb_cores_per_node

walltime = "00:30:00"
max_elapsed = "00:25:00"
type_fft = "'fluidfft.fft3d.mpi_with_pfft'"


for N in Ns:
    for Rb in Rbs:
        for proj in projs:
            if proj == "None":
                command = f'run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end 100 --max-elapsed {max_elapsed} --modify-params "params.oper.type_fft = {type_fft}"'
            else:
                command = f'run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end 100 --projection={proj} --max-elapsed {max_elapsed} --modify-params "params.oper.type_fft = {type_fft}"'
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
