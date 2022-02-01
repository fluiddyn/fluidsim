from fluiddyn.clusters.licallo import Licallo as Cluster

cluster = Cluster()

# Ns = [10, 20, 40]
# Rbs = [10, 20, 40]
# projs = [None, "poloidal"]

Ns = [40]
Rbs = [40]
projs = ['"poloidal"']

nz = 120
max_elapsed = "11:50:00"

for N in Ns:
    for Rb in Rbs:
        for proj in projs:
            cluster.submit_script(
                f"run_simul_polo.py -N {N} -Rb {Rb} -nz {nz} --t_end 20 --projection={proj} --spatiotemporal-spectra --max-elapsed {max_elapsed}",
                name_run=f"ns3d.strat_N{N}_Rb{Rb}",
                nb_cores_per_node=20,
                nb_mpi_processes=20,
                omp_num_threads=1,
                ask=True,
                walltime="12:00:00",
            )
