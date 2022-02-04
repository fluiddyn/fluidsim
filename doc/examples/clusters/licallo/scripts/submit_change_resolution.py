import sys
import os
import glob
from pathlib import Path
from fluidlicallo import cluster

# Parameters of the simulation
dir_path = "/scratch/vlabarre/aniso/"
Fh = 1./3.
Rb = 5
nz = 20
nh = 4*nz

paths = glob.glob(dir_path + f"ns3d.strat_polo_Fh{Fh:.3e}_Rb{Rb:.3g}_{nh:d}x{nh:d}x{nz}_" + '*')

nb_nodes = 1
nb_cores_per_node = 4 #cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node


for path in paths:
    print(path)
    title = f"Fh{Fh:.3e}_Rb{Rb:.3g}"

    if not cluster:
        continue

    print(f"doubling the resolution (sequential)\n")

    command = f"fluidsim-modif-resolution {path} 2"
    os.system(command)

    path_new_state = next(Path(path).glob("State_phys*"))

    command = (
        f'fluidsim-restart {path_new_state} --add-to-t_end 5.0 --modify-params "params.NEW_DIR_RESULTS = True; params.nu_2 /= 2; '
        f'params.output.periods_print.print_stdout = 0.05; params.output.periods_save.spatial_means = 0.01;"'        
    )


    print(f"Restart the simulation with doubled resolution and viscosity divided by two, submitting:\n {command}")

    cluster.submit_command(
        f"{command}",
        name_run=f"restart_reso_" + title,
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        nb_mpi_processes=nb_mpi_processes,
        omp_num_threads=1,
        ask=True,
        walltime="12:00:00",
    )

  
