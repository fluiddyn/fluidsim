import os
import subprocess
from pathlib import Path
from itertools import product
import re
from math import pi
from pprint import pprint
import sys

import pytimeparse
from capturer import CaptureOutput

from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)
from fluidoccigen import cluster

COMPUTE_SPATIOTEMP_SPECTRA = "--no-spatiotemp-spectra" not in sys.argv

path_scratch = Path(os.environ["SCRATCHDIR"])
path_base = path_scratch / "aniso"


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


couples1344 = set(
    lprod([10], [160])
    + lprod([20], [40, 80])
    + lprod([40], [10, 20, 40, 80])
    + lprod([80, 120], [10])
)
couples1792 = set(
    lprod([20], [40, 80]) + lprod([40], [10, 20, 40]) + lprod([80, 120], [10])
)
couples2240 = set(
    lprod([20], [80]) + lprod([40], [20, 40]) + lprod([80, 120], [10])
)


def submit_from_file(nh, nh_small, t_end, nb_nodes_from_N, type_fft_from_N):

    print("path_base:", path_base, sep="\n")

    paths_in = sorted(
        path_base.glob(
            f"ns3d.strat_toro*_{nh_small}x{nh_small}*/State_phys_{nh}x{nh}*"
        )
    )
    print("paths_in :")
    pprint([p.parent.name for p in paths_in])

    path_simuls = sorted(path_base.glob(f"ns3d.strat_toro*_{nh}x{nh}*"))
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


def submit_restart(nh, t_end, nb_nodes_from_N, max_elapsed_time="23:30:00"):

    nb_cores_per_node = 28
    max_elapsed_time = pytimeparse.parse(max_elapsed_time)

    print("path_base:", path_base, sep="\n")
    path_simuls = sorted(path_base.glob(f"ns3d.strat_toro*_{nh}x{nh}*"))

    jobs_id, jobs_name, jobs_runtime = get_info_jobs()

    for path_simul in path_simuls:
        t_start, t_last = times_start_last_from_path(path_simul)
        if t_last >= t_end:
            print(f"{path_simul.name:90s}: completed ({t_last=}).")
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


def postrun(t_end, nh, coef_modif_resol=None, couples_larger_resolution=None):
    """
    For each finished simulation:

    1. clean up the directory
    2. prepare a file with larger resolution
    3. compute the spatiotemporal spectra
    4. execute and save a notebook analyzing the simulation

    """

    import papermill as pm
    from fluiddyn.util import modification_date
    from fluidsim.util import times_start_last_from_path, load_params_simul
    from fluidsim import load

    if coef_modif_resol is not None:
        nh_larger = int(nh * eval(coef_modif_resol))

    deltat = 0.05

    print("path_base:", path_base, sep="\n")

    path_output_papermill = path_base / "results_papermill"
    path_output_papermill.mkdir(exist_ok=True)

    path_end_states = path_base / "end_states"
    path_end_states.mkdir(exist_ok=True)

    paths = sorted(path_base.glob(f"ns3d*_toro_*_{nh}x{nh}x*"))

    for path in paths:
        t_start, t_last = times_start_last_from_path(path)

        if t_last < t_end:
            try:
                estimated_remaining_duration = (
                    get_last_estimated_remaining_duration(path)
                )
            except RuntimeError:
                estimated_remaining_duration = "?"

            print(
                f"{path.name:90s} not finished ({t_last=}, {estimated_remaining_duration=})"
            )

            if (path / "is_being_advanced.lock").exists():
                continue

            # soft link of the last file (to be able to restart)
            path_files = sorted(path.glob("state_phys*"))
            if not path_files:
                continue

            path_last_state = path_files[-1]
            link_last_state = path_end_states / path.name / path_last_state.name
            if not link_last_state.exists():
                link_last_state.parent.mkdir(exist_ok=True)
                link_last_state.symlink_to(path_last_state)
                print(f"Link {link_last_state} created")
            continue

        params = load_params_simul(path)
        N = float(params.N)
        nx = params.oper.nx
        Rb = float(re.search(r"_Rb(.*?)_", path.name).group(1))

        tmp = f"{N=} {Rb=} {nh=}"
        print(f"{tmp:40s}: completed")

        # delete some useless restart files
        deltat_file = params.output.periods_save.phys_fields
        path_files = sorted(path.glob("state_phys*"))
        for path_file in path_files:
            time = float(path_file.name.rsplit("_t", 1)[1][:-3])
            if time % deltat_file > deltat:
                print(f"deleting {path_file.name}")
                path_file.unlink()

        path_end_state = path_file
        link_last_state = path_end_states / path.name / path_end_state.name
        if not link_last_state.exists():
            link_last_state.parent.mkdir(exist_ok=True)
            link_last_state.symlink_to(path_end_state)

        # compute spatiotemporal spectra
        sim = load(path, hide_stdout=True)
        t_statio = round(t_start) + 1.0
        if nh == 896 and N >= 80:
            t_statio += 4

        if COMPUTE_SPATIOTEMP_SPECTRA and not (nh in [896, 1344] and N >= 80):
            path_spatiotemp = path / "spatiotemporal"
            if list(path_spatiotemp.glob("rank*.h5")):
                sim.output.spatiotemporal_spectra.get_spectra(tmin=t_statio)

        if coef_modif_resol is not None:
            try:
                next(path.glob(f"State_phys_{nh_larger}x{nh_larger}*"))
            except StopIteration:
                if (N, Rb) in couples_larger_resolution:
                    subprocess.run(
                        f"fluidsim-modif-resolution {path} {coef_modif_resol}".split()
                    )

        path_in = "../analyse_1simul_papermill.ipynb"
        path_out = (
            path_output_papermill
            / f"analyze_N{N:05.2f}_Rb{Rb:03.0f}_nx{nx:04d}.ipynb"
        )

        date_in = modification_date(path_in)
        try:
            date_out = modification_date(path_out)
        except FileNotFoundError:
            has_to_run = True
        else:
            has_to_run = date_in > date_out

        if (
            has_to_run
            and COMPUTE_SPATIOTEMP_SPECTRA
            and not (nh in [896, 1344] and N >= 80)
        ):
            pm.execute_notebook(
                path_in, path_out, parameters=dict(path_dir=str(path))
            )
            print(f"{path_out} saved")


def nb_nodes_from_N_896(N):
    if N >= 80:
        return 2
    return 4


def nb_nodes_from_N_1344(N):
    if N >= 80:
        return 4
    elif N == 10:
        return 12
    return 8


def nb_nodes_from_N_1792(N):
    if N >= 80:
        return 8
    return 16


def nb_nodes_from_N_2240(N):
    if N >= 80:
        return 8
    return 16
