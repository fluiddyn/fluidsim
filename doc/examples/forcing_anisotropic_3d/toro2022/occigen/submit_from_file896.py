import re
from math import pi
from pprint import pprint

from fluidoccigen import cluster

from util import get_info_jobs, path_scratch, nb_nodes_from_N_896

t_end = 40.0
nh = 896
path_init = path_scratch / "2022/aniso/init_occigen"
print(f"{path_init = }")

paths_in = sorted(path_init.glob("ns3d.strat_toro*_640x640*"))
print("paths_in :")
pprint([p.name for p in paths_in])

path_simuls = sorted(
    (path_scratch / "aniso").glob(f"ns3d.strat_toro*_{nh}x{nh}*")
)
print("path_simuls:")
pprint([p.name for p in path_simuls])

jobs_id, jobs_name, jobs_runtime = get_info_jobs()
jobs_name = set(jobs_name.values())
print(f"{jobs_name=}")


def type_fft_from_N(N):
    if N >= 80:
        return "fftwmpi3d"
    else:
        return "fftw1d"


for path_init_dir in paths_in:

    name_old_sim = path_init_dir.name

    N_str = re.search(r"_N(.*?)_", name_old_sim).group(1)
    N = float(N_str)
    Rb_str = re.search(r"_Rb(.*?)_", name_old_sim).group(1)
    Rb = float(Rb_str)

    N_str = "_N" + N_str
    Rb_str = "_Rb" + Rb_str

    if [p for p in path_simuls if N_str in p.name and Rb_str in p.name]:
        print(f"Simulation directory for {N=} and {Rb=} already created")
        continue

    name_run = f"N{N}_Rb{Rb}_{nh}"
    if name_run in jobs_name:
        print(f"Job {name_run} already launched")
        continue

    path_init_file = next(
        path_init_dir.glob(f"State_phys_{nh}x{nh}*/state_phys*")
    )

    assert path_init_file.exists()
    print(path_init_file)

    period_spatiotemp = min(2 * pi / (N * 8), 0.03)

    type_fft = type_fft_from_N(N)
    nb_nodes = nb_nodes_from_N_896(N)

    command = (
        f"fluidsim-restart {path_init_file} --t_end {t_end} --new-dir-results "
        "--max-elapsed 23:30:00 "
        "--modify-params 'params.nu_4 /= 3.07; params.output.periods_save.phys_fields = 0.5; "
        f'params.oper.type_fft = "fft3d.mpi_with_{type_fft}"; '
        f"params.output.periods_save.spatiotemporal_spectra = {period_spatiotemp}'"
    )

    nb_cores_per_node = cluster.nb_cores_per_node
    nb_mpi_processes = nb_cores_per_node * nb_nodes

    print(f"Submitting command\n{command}")

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
