import re

import pytimeparse
from capturer import CaptureOutput

from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from fluidoccigen import cluster

from util import get_info_jobs, path_scratch

t_end = 44.0
nh = 1344
nb_cores_per_node = 28
max_elapsed_time = pytimeparse.parse("23:30:00")

path_simuls = sorted(
    (path_scratch / "aniso").glob(f"ns3d.strat_toro*_{nh}x{nh}*")
)

jobs_id, jobs_name, jobs_runtime = get_info_jobs()

for path_simul in path_simuls:
    t_start, t_last = times_start_last_from_path(path_simul)
    if t_last >= t_end:
        continue

    try:
        estimated_remaining_duration = get_last_estimated_remaining_duration(
            path_simul
        )
    except RuntimeError:
        print(
            "Cannot submit more jobs because no estimation of remaining duration"
        )
        continue

    N_str = re.search(r"_N(.*?)_", path_simul.name).group(1)
    N = float(N_str)
    Rb_str = re.search(r"_Rb(.*?)_", path_simul.name).group(1)
    Rb = float(Rb_str)

    name_run = f"N{N}_Rb{Rb}_{nh}"

    if N == 20:
        nb_nodes = 4
    else:
        nb_nodes = 8
    nb_mpi_processes = nb_nodes * nb_cores_per_node

    jobs_simul = {
        key: value for key, value in jobs_name.items() if value == name_run
    }
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
        continue
    # get number of jobs to be sent
    nb_jobs = (
        estimated_remaining_duration - time_submitted
    ) // max_elapsed_time + 1
    print(f"{nb_jobs} jobs need to be submitted")

    for i_job in range(nb_jobs):
        with CaptureOutput() as capturer:
            cluster.submit_command(
                command,
                name_run=name_run,
                nb_nodes=nb_nodes,
                nb_cores_per_node=nb_cores_per_node,
                nb_mpi_processes=nb_mpi_processes,
                omp_num_threads=1,
                ask=False,
                walltime="23:59:59",
                dependency=job_id,
            )
            text = capturer.get_text()

        job_id = text.split()[-1].strip()
