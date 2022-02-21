import sys
import os
import glob
from pathlib import Path
from fluidlicallo import cluster

# Parameters of the simulation
dir_path = f"/scratch/vlabarre/aniso/"
N = 40.
Fh = 1./N
Rb = 20
proj = "poloidal"
nz = 120
nh = 4 * nz
add_time = 10.0
type_fft = "'fluidfft.fft3d.mpi_with_fftwmpi3d'" #default  # Usefull when it is necessary to change the type of decomposition for fft (example: fftw3d -> p3dfft)
coef_reso = 4/3
coef_nu_4 = 1. / (coef_reso ** (10./3.))

nb_nodes = 2 
nb_cores_per_node = cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node
walltime = "23:55:00"


# Path of the simulation
if proj	== "None":
    paths = glob.glob(dir_path + f"ns3d.strat_polo_Fh{Fh:.3e}_Rb{Rb:.3g}_{nh:d}x{nh:d}x{nz:d}_" + '*')
elif proj == "poloidal":
    paths = glob.glob(dir_path + f"ns3d.strat_polo_proj_Fh{Fh:.3e}_Rb{Rb:.3g}_{nh:d}x{nh:d}x{nz:d}_" + '*')
else:
    print('Projection (variable proj) must be "None" or "poloidal"')
 


# Change resolution and hyperviscosity, then restart
for path in paths:
    print(path)

    if not cluster:
        continue
   
    print(f"Changing the resolution (sequential) \n")

    command = f"fluidsim-modif-resolution {path} {coef_reso}"

    os.system(command)
 
    """
    cluster.submit_command(
        f'{command}',
        name_run=f"ns3d.strat_proj{proj}_Fh{Fh:.3e}_Rb{Rb:.3g}",
        nb_nodes=1,
        nb_cores_per_node=1,
        nb_mpi_processes=1,
        omp_num_threads=1,
        ask=True,
        walltime="00:15:00",
        partition="x40",
        mem="10G",
    )
    """
    nh = int(coef_reso * nh)
    nz = int(coef_reso * nz)  
    path_new_state = next(Path(path).glob(f"State_phys_{nh:d}x{nh:d}x{nz:d}*"))
    print(path_new_state)

    command = (
        f'fluidsim-restart {path_new_state} --add-to-t_end {add_time} '
        f'--modify-params "params.NEW_DIR_RESULTS = True; '
        f'params.output.periods_print.print_stdout = 0.05; '
        f'params.nu_4 *= {coef_nu_4}; '
        f'params.oper.type_fft = {type_fft};"'  
    )

    print(command)

    print(f"Restart the simulation with the new resolution, submitting:\n {command}")

    cluster.submit_command(
        f'{command}',
        name_run=f"ns3d.strat_proj{proj}_Fh{Fh:.3e}_Rb{Rb:.3g}",
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        nb_mpi_processes=nb_mpi_processes,
        omp_num_threads=1,
        ask=True,
        walltime=walltime,
        dependency="singleton",
    )
    
