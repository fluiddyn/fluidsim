from fluidlicallo import cluster

# Ns = [100]
# Rbs = [5, 10, 20, 40]
# projs = ["None", "poloidal"]

Ns = [100]
Rbs = [5, 10, 20]
projs = ["None", "poloidal"]
nz = 160

nb_nodes = 2
nb_cores_per_node = cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node

walltime = "23:55:00"
max_elapsed = "23:45:00"
type_fft = "'fluidfft.fft3d.mpi_with_fftwmpi3d'"


for N in Ns:
    Fh = 1./N
    for Rb in Rbs:
        for proj in projs:
            if proj == "None":
                t_end = 20.
                command = f'./run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end {t_end} --max-elapsed {max_elapsed} --modify-params "params.oper.type_fft = {type_fft};"'
                #command = f'./run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end {t_end} --max-elapsed {max_elapsed}'
            elif proj == "poloidal":
                t_end = 5.
                command = f'./run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end {t_end} --projection={proj} --max-elapsed {max_elapsed} --modify-params "params.oper.type_fft = {type_fft};"'
                #command = f'./run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end {t_end} --projection={proj} --max-elapsed {max_elapsed}'
            else:
                print('Projection (variable proj) must be "None" or "poloidal"')
 
            cluster.submit_command(
                f"{command}",
                name_run=f"ns3d.strat_proj{proj}_Fh{Fh:.3e}_Rb{Rb:.3g}",
                nb_nodes=nb_nodes,
                nb_cores_per_node=nb_cores_per_node,
                nb_mpi_processes=nb_mpi_processes,
                omp_num_threads=1,
                ask=True,
                walltime=walltime,
            )
