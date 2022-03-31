import os
import subprocess
from pathlib import Path
from itertools import product
import re
from math import pi
from pprint import pprint

import pytimeparse
from capturer import CaptureOutput

from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)
from fluidoccigen import cluster

path_scratch = Path(os.environ["SCRATCHDIR"])


def run(command):
    return subprocess.run(
        command.split(), check=True, capture_output=True, text=True
    )


user = os.environ["USER"]


def get_info_jobs():

    process = run(f"squeue -u {user}")
    lines = process.stdout.split("\n")[1:]
    jobs_id = [line.split()[0] for line in lines if line]

    jobs_name = {}
    jobs_runtime = {}
    for job_id in jobs_id:
        process = run(f"scontrol show job {job_id}")
        for line in process.stdout.split("\n"):
            line = line.strip()
            if line.startswith("JobId"):
                job_name = line.split(" JobName=")[1]
                jobs_name[job_id] = job_name
            elif line.startswith("RunTime"):
                jobs_runtime[job_id] = line.split("RunTime=")[1].split(" ")[0]

    return jobs_id, jobs_name, jobs_runtime


def lprod(a, b):
    return list(product(a, b))


def submit_restart(nh, t_end, nb_nodes_from_N, max_elapsed_time="23:30:00"):

    nb_cores_per_node = 28
    max_elapsed_time = pytimeparse.parse(max_elapsed_time)

    path_simuls = sorted(
        (path_scratch / "aniso").glob(f"ns3d.strat_toro*_{nh}x{nh}*")
    )

    jobs_id, jobs_name, jobs_runtime = get_info_jobs()

    for path_simul in path_simuls:
        t_start, t_last = times_start_last_from_path(path_simul)
        if t_last >= t_end:
            print(f"simulation {path_simul} finished ({t_last=}).")
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

        nb_nodes = nb_nodes_from_N(N)
        nb_mpi_processes = nb_nodes * nb_cores_per_node

        jobs_simul = {
            key: value for key, value in jobs_name.items() if value == name_run
        }
        jobs_runtime_simul = {key: jobs_runtime[key] for key in jobs_simul}

        print(
            f"{path_simul.name}: {t_last = }, {estimated_remaining_duration = }"
        )
        print(f"{jobs_simul}")

        # convert in seconds
        estimated_remaining_duration = pytimeparse.parse(
            estimated_remaining_duration
        )

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


def submit_from_file(nh, nh_small, t_end, nb_nodes_from_N, type_fft_from_N):

    paths_in = sorted(
        path_scratch.glob(
            f"aniso/ns3d.strat_toro*_{nh_small}x{nh_small}*/State_phys_{nh}x{nh}*"
        )
    )
    print("paths_in :")
    pprint([p.parent.name for p in paths_in])

    path_simuls = sorted(
        (path_scratch / "aniso").glob(f"ns3d.strat_toro*_{nh}x{nh}*")
    )
    print("path_simuls:")
    pprint([p.name for p in path_simuls])

    jobs_id, jobs_name, jobs_runtime = get_info_jobs()
    jobs_name = set(jobs_name.values())

    for path_init_dir in paths_in:
        path_init_dir = path_init_dir.parent
        name_old_sim = path_init_dir.name

        N_str = re.search(r"_N(.*?)_", name_old_sim).group(1)
        N = float(N_str)
        Rb_str = re.search(r"_Rb(.*?)_", name_old_sim).group(1)
        Rb = float(Rb_str)

        type_fft = type_fft_from_N(N)
        nb_nodes = nb_nodes_from_N(N)

        N_str = "_N" + N_str
        Rb_str = "_Rb" + Rb_str

        if [p for p in path_simuls if N_str in p.name and Rb_str in p.name]:
            print(f"Simulation directory for {N=} and {Rb=} already created")
            continue

        name_run = f"N{N}_Rb{Rb}_{nh}"
        if name_run in jobs_name:
            print(f"Job {name_run} already submitted")
            continue

        path_init_file = next(
            path_init_dir.glob(f"State_phys_{nh}x{nh}*/state_phys*")
        )

        assert path_init_file.exists()
        print(path_init_file)

        period_spatiotemp = min(2 * pi / (N * 8), 0.03)

        coef_decrease_nu4 = (nh / nh_small) ** (10 / 3)

        command = (
            f"fluidsim-restart {path_init_file} --t_end {t_end} --new-dir-results "
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


def nb_nodes_from_N_1344(N):
    if N == 20:
        return 4
    else:
        return 8


def nb_nodes_from_N_1792(N):
    return 16
