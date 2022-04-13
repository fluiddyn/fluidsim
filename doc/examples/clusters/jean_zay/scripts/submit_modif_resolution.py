import sys
import os
import glob
from pathlib import Path
from fluidjean_zay import cluster

# Parameters of the simulation
dir_path = f"/gpfswork/rech/zmr/uey73qw/aniso/"
N = 100.0
Fh = 1.0 / N
Rb = 20
proj = "None"
nz = 160
nh = 4 * nz
add_time = 5.0
type_fft = "'fluidfft.fft3d.mpi_with_pfft'"  # default  # Usefull when it is necessary to change the type of decomposition for fft (example: fftw3d -> p3dfft)
coef_reso = 2
coef_nu_4 = 1.0 / (coef_reso ** (10.0 / 3.0))

# Parameters of the job
nb_nodes = 8
nb_cores_per_node = cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node
walltime = "11:55:00"

# Find the path of the simulation
dir_path += "ns3d.strat_polo_"
if proj == "poloidal":
    dir_path += "proj_"
dir_path += f"Fh{Fh:.3e}_Rb{Rb:.3g}_{nh:d}x{nh:d}x{nz:d}_*"
print(dir_path)
paths = sorted(glob.glob(dir_path))
path = paths[-1]
print(path)

# Change resolution and hyperviscosity

"""
print(f"Changing the resolution (sequential) \n")

command = f"fluidsim-modif-resolution {path} {coef_reso}"
#os.system(command)

cluster.submit_command(
    f'{command}',
    name_run=f"ns3d.strat_proj{proj}_Fh{Fh:.3e}_Rb{Rb:.3g}",
    nb_nodes=1,
    nb_cores_per_node=1,
    nb_mpi_processes=1,
    omp_num_threads=1,
    ask=True,
    walltime="00:15:00",
    #partition="prepost", # Does not work if we use prepost
    mem="20Go",
)

"""

nh = int(coef_reso * nh)
nz = int(coef_reso * nz)
print(path + f"/State_phys_{nh:d}x{nh:d}x{nz:d}" + "*")
path_new_state = glob.glob(path + f"/State_phys_{nh:d}x{nh:d}x{nz:d}" + "*")
new_path = path_new_state[0]

command = (
    f"fluidsim-restart {new_path} --add-to-t_end {add_time} "
    f"--new-dir-results "
    f"--merge-missing-params "
    f'--modify-params "'
    f"params.oper.type_fft = {type_fft}; "
    f"params.output.periods_print.print_stdout = 0.05; "
    f"params.nu_4 *= {coef_nu_4}; "
    f'"'
)

print(f"Restart the simulation with the new resolution, submitting:\n {command}")

cluster.submit_command(
    f"{command}",
    name_run=f"ns3d.strat_proj{proj}_Fh{Fh:.3e}_Rb{Rb:.3g}",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node,
    nb_mpi_processes=nb_mpi_processes,
    omp_num_threads=1,
    ask=True,
    walltime=walltime,
    dependency="singleton",
)
