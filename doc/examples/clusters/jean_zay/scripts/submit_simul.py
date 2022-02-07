from fluidjean_zay import cluster

# Ns = [10, 20, 40]
# Rbs = [10, 20, 40]
# projs = ["None", "poloidal"]

Ns = [40]
Rbs = [40]
projs = ["poloidal"]
nz = 160

nb_nodes = 2
nb_cores_per_node = cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node

walltime = "03:00:00"
max_elapsed = "02:50:00"

for N in Ns:
    for Rb in Rbs:
        for proj in projs:
            if proj == "None":
                command = f"run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end 100 --max-elapsed {max_elapsed}"
            else:
                command = f"run_simul.py -N {N} -Rb {Rb} -nz {nz} --t_end 100 --projection={proj} --max-elapsed {max_elapsed}"
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
