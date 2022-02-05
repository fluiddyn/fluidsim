import sys
import glob
from pathlib import Path
from fluidlicallo import cluster

# Parameters of the simulation 
dir_path = "/scratch/vlabarre/aniso/"
Fh = 0.01
Rb = 40
nz = 120
nh = 480
add_time = 40.0

paths = glob.glob(dir_path + f"ns3d.strat_polo_Fh{Fh:.3e}_Rb{Rb:.3g}_{nh:d}x{nh:d}x{nz:d}_" + '*')

nb_nodes = 1
nb_cores_per_node = cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node


for path in paths:
    command = f"fluidsim-restart {path} --add-to-t_end {add_time}"

    print(f"submitting:\npython {command}")
    title = f"Fh{Fh:.3e}_Rb{Rb:.3g}"

    if not cluster:
        continue

    cluster.submit_command(
        f'{command}',
        name_run=f"restart_" + title,
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        nb_mpi_processes=nb_mpi_processes,
        omp_num_threads=1,
        ask=True,
        walltime="12:00:00",
        dependency="singleton",
    )

  
