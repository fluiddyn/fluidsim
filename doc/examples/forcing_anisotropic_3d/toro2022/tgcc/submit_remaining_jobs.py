import sys

import pytimeparse
from capturer import CaptureOutput

from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from util import (
    parse_args,
    get_sim_info_from_args,
    cluster,
    get_info_jobs,
    path_base,
    nb_nodes_from_nhnz,
)

max_elapsed_time = pytimeparse.parse("23:30:00")

args = parse_args()

sim = get_sim_info_from_args(args)
print(sim)

nh = sim.nh
nz = sim.nz

print("path_base:", path_base, sep="\n")
path_simuls = sorted(
    path_base.glob(f"ns3d.strat_toro*_Rb{args.Rb:.3g}_{nh}x{nh}*_N{args.N}_*")
)

try:
    path_simul = path_simuls[0]
except IndexError:
    print(
        f"Cannot find the simulation directory corresponding to {args.__dict__}"
    )
    sys.exit()

t_start, t_last = times_start_last_from_path(path_simul)
if t_last >= sim.t_end:
    print(f"{path_simul.name:90s}: completed ({t_last=}).")
    sys.exit()

try:
    estimated_remaining_duration = get_last_estimated_remaining_duration(
        path_simul
    )
except RuntimeError:
    print("Cannot submit more jobs because no estimation of remaining duration")
    sys.exit()

name_run = f"N{args.N}_Rb{args.Rb}_{args.nh}"

nb_nodes = nb_nodes_from_nhnz(nh, nz)
nb_mpi_processes = nb_nodes * cluster.nb_cores_per_node

jobs_id, jobs_name, jobs_runtime = get_info_jobs()
jobs_simul = {key: value for key, value in jobs_name.items() if value == name_run}
jobs_runtime_simul = {key: jobs_runtime[key] for key in jobs_simul}

print(f"{path_simul.name}: {t_last = }, {estimated_remaining_duration = }")
print(f"{jobs_simul}")

# convert in seconds
estimated_remaining_duration = pytimeparse.parse(estimated_remaining_duration)

print(f"{estimated_remaining_duration = }")

command = f"fluidsim-restart {path_simul} --modify-params 'params.output.periods_print.print_stdout = 0.02'"

# get last job id
try:
    job_id = max(jobs_simul.keys())
except ValueError:
    job_id = None

# get second remaining from jobs
time_submitted = max_elapsed_time * len(jobs_simul) - sum(
    pytimeparse.parse(runtime) for runtime in jobs_runtime_simul.values()
)
print(f"{time_submitted = }")

if estimated_remaining_duration < time_submitted:
    sys.exit()
# get number of jobs to be sent
nb_jobs = (estimated_remaining_duration - time_submitted) // max_elapsed_time + 1
print(f"{nb_jobs} jobs need to be submitted")

for i_job in range(nb_jobs):
    with CaptureOutput() as capturer:
        cluster.submit_command(
            command,
            name_run=name_run,
            nb_nodes=nb_nodes,
            nb_cores_per_node=cluster.nb_cores_per_node,
            nb_mpi_processes=nb_mpi_processes,
            omp_num_threads=1,
            ask=False,
            walltime="23:59:59",
            dependency=job_id,
        )
        text = capturer.get_text()

    job_id = text.split()[-1].strip()
