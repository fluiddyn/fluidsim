import sys
import re
from math import pi
from pprint import pprint

from util import (
    parse_args,
    get_sim_info_from_args,
    path_init_tgcc,
    cluster,
    get_info_jobs,
    path_base,
    type_fft_from_nhnz,
    nb_nodes_from_nhnz,
)

args = parse_args()

sim = get_sim_info_from_args(args)

print(sim)

nh = sim.nh
nz = sim.nz

if sim.init.startswith("("):
    raise NotImplementedError
else:

    path_simul_init = next(path_init_tgcc.glob(f"*{sim.name}*"))
    print(path_simul_init)

    try:
        path_init_file = next(
            path_simul_init.glob(f"State_phys_{nh}x{nh}x{nz}*/state_phys*")
        )
    except StopIteration:
        print("First run the modif_resolutions_legi.py script")
        sys.exit(0)

    name_old_sim = sim.init

assert path_init_file.exists()
print(path_init_file)

N_str = re.search(r"_N(.*?)_", name_old_sim).group(1)
N = args.N
Rb_str = re.search(r"_Rb(.*?)_", name_old_sim).group(1)
Rb = args.Rb

type_fft = type_fft_from_nhnz(nh, nz)
nb_nodes = nb_nodes_from_nhnz(nh, nz)

path_simuls = sorted(path_base.glob(f"ns3d.strat_toro*_{nh}x{nh}*"))
print("path_simuls:")
pprint([p.name for p in path_simuls])

N_str = "_N" + N_str
Rb_str = "_Rb" + Rb_str

if [p for p in path_simuls if N_str in p.name and Rb_str in p.name]:
    print(f"Simulation directory for {N=} and {Rb=} already created")
    sys.exit(0)


jobs_id, jobs_name, jobs_runtime = get_info_jobs()
jobs_name = set(jobs_name.values())

name_run = f"N{N}_Rb{Rb}_{nh}"
if name_run in jobs_name:
    print(f"Job {name_run} already submitted")
    sys.exit(0)

period_spatiotemp = min(2 * pi / (N * 8), 0.03)

coef_decrease_nu4 = (nh / sim.nh_small) ** (10 / 3)

command = (
    f"fluidsim-restart {path_init_file} --t_end {sim.t_end} --new-dir-results "
    "--max-elapsed 23:30:00 "
    f"--modify-params 'params.nu_4 /= {coef_decrease_nu4}; "
    "params.output.periods_save.phys_fields = 0.5; "
    f'params.oper.type_fft = "fft3d.mpi_with_{type_fft}"; '
    f"params.output.periods_save.spatiotemporal_spectra = {period_spatiotemp}'"
)

nb_cores_per_node = cluster.nb_cores_per_node
nb_mpi_processes = nb_cores_per_node * nb_nodes

print(f"Submitting command ({nb_nodes=})\n{command}")

cluster.submit_command(
    command,
    name_run=f"N{N}_Rb{Rb}_{nh}",
    nb_nodes=nb_nodes,
    nb_cores_per_node=nb_cores_per_node,
    nb_mpi_processes=nb_mpi_processes,
    omp_num_threads=1,
    ask=False,
    walltime="23:59:59",
)
