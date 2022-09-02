from util import *
from fluidlicallo import cluster

proj = "None"  # "poloidal"
N = 3
Rb = 3
type_fft = "'fluidfft.fft3d.mpi_with_fftwmpi3d'"

# assert N in [10, 20, 40]
# assert R in [5, 10, 20, 40, 80, 160]

nh = 320
ratio_nh_nz = get_ratio_nh_nz(N)

nz = nh // ratio_nh_nz

command = (
    f"./run_simul_polo.py -Rb {Rb} -N {N} --ratio-nh-nz {ratio_nh_nz} -nz {nz} "
    f"--max-elapsed 23:50:00 "
    f'--modify-params "'
    f"params.oper.type_fft = {type_fft}; "
    f'" '
)
if proj == "poloidal":
    command.append(f"--projection={proj}")

cluster.submit_command(
    f"{command}",
    name_run=f"proj{proj}_Rb{Rb}_N{N}_nh{nh}_nz{nz}",
    nb_nodes=1,
    walltime="23:59:30",
    nb_mpi_processes=4,  # cluster.nb_cores_per_node // 2,
    omp_num_threads=1,
    delay_signal_walltime=300,
    ask=True,
)
