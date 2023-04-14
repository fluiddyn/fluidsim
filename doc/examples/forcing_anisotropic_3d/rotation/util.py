from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
    load_params_simul,
)

from pathlib import Path
import subprocess
import os
from math import pi

from fluidjean_zay import cluster

path_base_licallo = Path("/scratch/vlabarre/aniso_rotation/")
path_base_azzurra = Path("/workspace/vlabarre/aniso_rotation/")
path_base_jeanzay = Path("/gpfsscratch/rech/uzc/uey73qw/aniso_rotation/")

path_base = path_base_jeanzay

coef_nu = 2.0
n_target = [320, 640, 1280]
Ro_target = [1.0, 10 ** (-1), 10 ** (-1.5), 10 ** (-2)]
walltime = "19:59:59"


def list_paths(Ro, n, NO_GEOSTROPHIC_MODES=False):
    # Find the paths of the simulations
    paths = sorted(path_base.glob(f"ns3d_polo*_{n}x{n}x{n}*"))
    pathstemp = [p for p in paths if f"_Ro{Ro:.3e}_" in p.name]

    if NO_GEOSTROPHIC_MODES:
        paths = [p for p in pathstemp if f"_NO_GEOSTROPHIC_MODES_" in p.name]
    else:
        paths = [p for p in pathstemp if f"_NO_GEOSTROPHIC_MODES_" not in p.name]

    print(
        f"List of paths for simulations with (Ro, n, NO_GEOSTROPHIC_MODES)= ({Ro:.3e}, {n}, {NO_GEOSTROPHIC_MODES}): \n"
    )

    for path in paths:
        print(path, "\n")

    return paths


def type_fft_from_n(n):
    "Get the fft type to use for a given n"
    if n == 320:
        return "fftwmpi3d"
    elif n == 640:
        return "pfft"
    elif n == 1280:
        return "pfft"
    else:
        raise NotImplementedError


def nb_nodes_from_n(n):
    if n == 320:
        return 4
    if n == 640:
        return 16
    if n == 1280:
        return 64


def max_elapsed_from_n(n):
    if n == 320:
        return "19:30:00"
    if n == 640:
        return "19:00:00"
    if n == 1280:
        return "18:30:00"


def get_t_statio(n, Ro):
    "Get stationarity time of the simulation with resolution n"
    if n == 160:
        return 0.0
    elif n == 320:
        return 150.0
    elif n == 640:
        return 170.0
    elif n == 1280:
        return 180.0
    else:
        raise NotImplementedError


def get_t_end(n, Ro):
    "Get end time of the simulation with resolution n"
    t_statio = get_t_statio(n, Ro)
    return t_statio + 10.0


def is_job_submitted(name_run):
    command = f"squeue -n {name_run}"
    out = subprocess.check_output(command, shell=True)
    length = len(out.splitlines())
    if length > 1:
        # In this case squeue's output contain at least one job
        return True
    else:
        return False


def submit(n=320, Ro=1e-1, NO_GEOSTROPHIC_MODES=False):
    t_statio = get_t_statio(n, Ro)
    t_end = get_t_end(n, Ro)
    nb_nodes = nb_nodes_from_n(n)
    nb_cores_per_node = cluster.nb_cores_per_node
    nb_mpi_processes = nb_cores_per_node * nb_nodes
    max_elapsed = max_elapsed_from_n(n)
    n_lower = n // 2
    t_statio_lower = get_t_statio(n_lower, Ro)
    type_fft = type_fft_from_n(n)

    params = f"{Ro=} {n=} {NO_GEOSTROPHIC_MODES=}"

    name_run = (
        f"run_simul_polo_Ro{Ro}_n{n}_NO_GEOSTROPHIC_MODES{NO_GEOSTROPHIC_MODES}"
    )
    path_runs = list_paths(Ro, n, NO_GEOSTROPHIC_MODES=NO_GEOSTROPHIC_MODES)
    path_runs_lower = list_paths(
        Ro, n_lower, NO_GEOSTROPHIC_MODES=NO_GEOSTROPHIC_MODES
    )

    if is_job_submitted(name_run):
        print(
            f"Nothing to do for {params} because first job is " "already launched"
        )
        return

    if len(path_runs) == 0:
        if n == 320:
            command = (
                f"./run_simul_polo.py --Ro {Ro} -n {n} -coef_nu {coef_nu} --t_end {t_statio} "
                f"--max-elapsed {max_elapsed} "
                f"--modify-params '"
                f'params.oper.type_fft = "fft3d.mpi_with_{type_fft}"; '
                f"'"
            )
            if NO_GEOSTROPHIC_MODES:
                command += f" --NO_GEOSTROPHIC_MODES {NO_GEOSTROPHIC_MODES}"

            cluster.submit_command(
                command,
                name_run=name_run,
                nb_nodes=nb_nodes,
                walltime=walltime,
                nb_mpi_processes=nb_mpi_processes,
                omp_num_threads=1,
                delay_signal_walltime=300,
                ask=True,
            )

        else:
            # We must restart from lower resolution
            if len(path_runs_lower) == 0:
                print(
                    f"Cannot do anything for {params} because no init directory. Need to run a simulation at smaller resolution."
                )
                return

            elif len(path_runs_lower) == 1:
                t_start_lower, t_last_lower = times_start_last_from_path(
                    path_runs_lower[0]
                )
                if t_last_lower < t_statio_lower:
                    try:
                        estimated_remaining_duration = (
                            get_last_estimated_remaining_duration(
                                path_runs_lower[0]
                            )
                        )
                    except RuntimeError:
                        estimated_remaining_duration = "?"

                    print(
                        f"Cannot launch {name_run} because the coarse "
                        "simulation is not finished\n"
                        f"  ({t_last_lower=} < {t_statio_lower=}, {estimated_remaining_duration = })"
                    )

                else:
                    path_state_lower = sorted(
                        path_runs_lower[0].glob(
                            f"State_phys_{n}x{n}x{n}*/state_phys_t*.nc"
                        )
                    )

                    if len(path_state_lower) == 0:
                        print(f"We change resolution for {path_runs_lower[0]}")
                        modif_reso(path=path_runs_lower[0], n=n_lower, Ro=Ro)
                    elif len(path_state_lower) == 1:
                        coef_change_reso = n / n_lower
                        coef_reduce_nu = coef_change_reso ** (4 / 3)
                        command = (
                            f"fluidsim-restart {path_runs_lower[0]} --t_end {t_statio} --new-dir-results "
                            f"--max-elapsed {max_elapsed} "
                            f"--modify-params '"
                            f"params.nu_2 /= {coef_reduce_nu}; "
                            f'params.oper.type_fft = "fft3d.mpi_with_{type_fft}"; '
                            f"params.output.periods_save.spatiotemporal_spectra = 0.0; "
                            f"'"
                        )
                        print(f"run: {command} \n")
                        cluster.submit_command(
                            command,
                            name_run=name_run,
                            nb_nodes=nb_nodes,
                            walltime=walltime,
                            nb_mpi_processes=nb_mpi_processes,
                            omp_num_threads=1,
                            delay_signal_walltime=300,
                            ask=True,
                        )
                    else:
                        print(
                            f"More than one state files in {path_runs_lower[0]}"
                        )
            else:
                print(
                    f"Zero or more than one init directory with {params})",
                    f"Nothing is done",
                )

    elif len(path_runs) == 1:
        t_start, t_last = times_start_last_from_path(path_runs[0])
        tmp = load_params_simul(path_runs[0])
        f = float(tmp.f)
        command = None
        if t_last < t_statio:
            print("we restart without saving spatiotemporal spectra")
            command = (
                f"fluidsim-restart {path_runs[0]} --t_end {t_statio} --max-elapsed {max_elapsed} "
                f"--modify-params '"
                f"params.output.periods_save.spatiotemporal_spectra = 0.0; "
                f"'"
            )
        elif t_last < t_end:
            period_spatiotemp = min(2 * pi / (f * 8), 0.03)
            iksmax = int(32 * n // 320)
            print("we restart and save spatiotemporal spectra")
            command = (
                f"fluidsim-restart {path_runs[0]} --t_end {t_end} --max-elapsed {max_elapsed} "
                f"--modify-params '"
                f"params.output.periods_save.spatiotemporal_spectra = {period_spatiotemp}; "
                f"params.output.spatiotemporal_spectra.probes_region = ({iksmax}, {iksmax}, {iksmax}); "
                f"'"
            )
        else:
            print(f"{params:40s}: completed")

        if command:
            try:
                estimated_remaining_duration = (
                    get_last_estimated_remaining_duration(path_runs[0])
                )
            except RuntimeError:
                estimated_remaining_duration = "?"

            print(
                f"{path_runs[0].name}: {t_last=}, {estimated_remaining_duration=}"
            )
            # Remove is_being_advanced.lock file
            try:
                path_file_to_remove = next(
                    path_runs[0].glob(f"is_being_advanced.lock")
                )
                path_file_to_remove.unlink()
            except StopIteration:
                print("No file to remove before launching the simulation")

            print(f"run: {command} \n")
            cluster.submit_command(
                command,
                name_run=name_run,
                nb_nodes=nb_nodes,
                walltime=walltime,
                nb_mpi_processes=nb_mpi_processes,
                omp_num_threads=1,
                delay_signal_walltime=300,
                ask=True,
                dependency="singleton",
            )

    else:
        print(
            f"More than one simulation with "
            f"{params:40s} \t"
            f"Nothing is done"
        )


def modif_reso(path, n, Ro, coef_change_reso=2):
    name_run = f"modif_reso_polo_{path}"
    t_statio = get_t_statio(n, Ro)
    command = f"fluidsim-modif-resolution {path} {coef_change_reso} --t_approx {t_statio}"
    print(f"run command: {command}\n")

    os.system(command)
    """
    # On Jean-Zay
    cluster.submit_command(
        f"{command}",
        name_run=name_run,
        nb_nodes=1,
        nb_cores_per_node=20,
        nb_cpus_per_task=20,
        nb_tasks_per_node=1,
        nb_tasks=1,
        nb_mpi_processes=1,
        omp_num_threads=1,
        ask=True,
        walltime="00:15:00",
        project="uzc@cpu",
        partition="prepost",
    )
    """
