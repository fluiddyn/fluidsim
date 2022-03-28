import os
from pathlib import Path
import re
from math import pi

from fluidoccigen import cluster

t_end = 40.0
nh = 896
nb_nodes = 4
path_scratch = Path(os.environ["SCRATCHDIR"])
path_init = path_scratch / "2022/aniso/init_occigen"

path_init_dir = (
    path_init
    / "ns3d.strat_toro_Fh2.500e-02_Rb80_640x640x80_V3x3x0.375_N40_2022-03-25_20-41-40"
)
name_old_sim = path_init_dir.name
N = float(re.search(r"_N(.*?)_", name_old_sim).group(1))
Rb = float(re.search(r"_Rb(.*?)_", name_old_sim).group(1))

path_init_file = next(path_init_dir.glob(f"State_phys_{nh}x{nh}*/state_phys*"))

assert path_init_file.exists()
print(path_init_file)

period_spatiotemp = min(2 * pi / (N * 8), 0.03)

command = (
    f"fluidsim-restart {path_init_file} --t_end {t_end} --new-dir-results "
    "--max-elapsed 23:50:00 "
    "--modify-params 'params.nu_4 /= 3.07; params.output.periods_save.phys_fields = 0.5; "
    'params.oper.type_fft = "fft3d.mpi_with_fftw1d"; '
    f"params.output.periods_save.spatiotemporal_spectra = {period_spatiotemp}'"
)

nb_cores_per_node = cluster.nb_cores_per_node
nb_mpi_processes = nb_cores_per_node * nb_nodes

cluster.submit_command(
    command,
    name_run=f"N{N}_Rb{Rb}_{nh}",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node,
    nb_mpi_processes=nb_mpi_processes,
    omp_num_threads=1,
    ask=False,
    walltime="23:59:58",
)
