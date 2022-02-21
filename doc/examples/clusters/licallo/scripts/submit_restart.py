import sys
import glob
from pathlib import Path
from fluidlicallo import cluster
from math import pi

# Parameters of the simulation 
dir_path = "/scratch/vlabarre/aniso/"
N = 100.
Fh = 1./N
Rb = 40
proj = "poloidal"
nz = 160
nh = 4 * nz
add_time = 10.0

nb_nodes = 2
nb_cores_per_node = cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node

walltime = "23:55:00"
type_fft = "'fluidfft.fft3d.mpi_with_fftwmpi3d'"


# Path of the simulation
if proj == "None":
    paths = glob.glob(dir_path + f"ns3d.strat_polo_Fh{Fh:.3e}_Rb{Rb:.3g}_{nh:d}x{nh:d}x{nz:d}_" + '*')
elif proj == "poloidal":
    paths = glob.glob(dir_path + f"ns3d.strat_polo_proj_Fh{Fh:.3e}_Rb{Rb:.3g}_{nh:d}x{nh:d}x{nz:d}_" + '*')
else:
    print('Projection (variable proj) must be "None" or "poloidal"')

print(paths)

assert len(paths) <= 1, "More than one simulation with N={N}, Rb={Rb}, proj={proj}, nz={nz}, nh={nh}"

# Restart
for path in paths:
    period_save_spatiotemporal_spectra = 2 * pi / (8 * N)
    command = ( 
        f'fluidsim-restart {path} --add-to-t_end {add_time} '
        f'--modify-params "params.oper.type_fft = {type_fft}; '
        f'params.output.periods_save.spatiotemporal_spectra = {period_save_spatiotemporal_spectra};"'
    )

    print(f"submitting:\npython {command}")

    if not cluster:
        continue

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

  
