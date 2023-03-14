from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from pathlib import Path
import subprocess

from fluidlicallo import cluster

path_base_licallo = Path("/scratch/vlabarre/rotation/")
path_output_papermill_licallo = Path("/scratch/vlabarre/rotation/results_papermill")

path_base_azzurra = Path("/workspace/vlabarre/rotation/")
path_output_papermill_azzurra = Path(
    "/workspace/vlabarre/rotation/results_papermill"
)

path_base_jeanzay = Path("/gpfsscratch/rech/uzc/uey73qw/rotation/")
path_output_papermill_jeanzay = Path(
    "/gpfsscratch/rech/uzc/uey73qw/rotation/results_papermill"
)

path_base = path_base_licallo
path_output_papermill = path_output_papermill_licallo

coef_nu = 1.2
n_target = [320, 640]
Ro_target = [1.0, 10**(-0.5), 10**(-1), 10**(-1.5), 10**(-2)]
walltime = "19:59:59"

def list_paths(Ro, n, NO_GEOSTROPHIC_MODES=False):
    # Find the paths of the simulations
    paths = sorted(path_base.glob(f"ns3d.strat_polo*_{n}x{n}x{n}*"))

    pathstemp = [
        p for p in paths if f"_Ro{Ro:.3e}_" in p.name 
    ]

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
    return "fftw1d"
    

def nb_nodes_from_n(n):
    if n == 320:
        return 1
    if n == 640:
        return 4
    if n == 1280:
        return 32
    
def max_elapsed_from_n(n):
    if n == 320:
        return "19:40:00" 
    if n == 640:
        return "19:00:00"
    if n == 1280:
        return "18:30:00"

def get_t_end(n):
    "Get end time of the simulation with resolution n"
    if n == 320:
        return 20.0
    elif n == 640:
        return 30.0
    elif n == 1280:
        return 35.0
    else:
        raise NotImplementedError


def get_t_statio(f, n):
    "Get end time of the simulation with resolution n"
    t_end = get_t_end(n) - 2.0

def is_job_submitted(name_run):
    command = f"squeue -n {name_run}"
    out = subprocess.check_output(command, shell=True)
    length = len(out.splitlines())
    if length > 1:
        # In this case squeue's output contain at least one job
        return True
    else:
        return False

def submit(n=320,Ro=1e-1,NO_GEOSTROPHIC_MODES=False):
    t_end = get_t_end(n)
    nb_nodes = nb_nodes_from_n(n)
    nb_cores_per_node = cluster.nb_cores_per_node
    nb_mpi_processes = nb_cores_per_node* nb_nodes
    max_elapsed = max_elapsed_from_n(n)

    params = f"{Ro=} {n=} {NO_GEOSTROPHIC_MODES=}"
    
    name_run = f"run_simul_polo_Ro{Ro}_n{n}_NO_GEOSTROPHIC_MODES{NO_GEOSTROPHIC_MODES}"
    path_runs = list_paths(Ro, n, NO_GEOSTROPHIC_MODES=False)

    if is_job_submitted(name_run):
        print(
            f"Nothing to do for Ro{Ro}_n{n}_NO_GEOSTROPHIC_MODES{NO_GEOSTROPHIC_MODES} because first job is "
            "already launched"
        )
        break


    if len(path_runs) == 0:
        command = (
            f"./run_simul_polo.py --Ro {Ro} -n {n} -coef_nu {coef_nu} --t_end {t_end} "
            f"--max-elapsed {max_elapsed} "
        )
        if NO_GEOSTROPHIC_MODES:
            command.append(f"--NO_GEOSTROPHIC_MODES {NO_GEOSTROPHIC_MODES}")

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

    elif len(path_runs) == 1:
        t_start, t_last = times_start_last_from_path(path_runs[0])
        if t_last >= t_end:
            print(f"{params:40s}: completed")
            break

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

        print("we restart")
        command = f"fluidsim-restart {path_runs[0]} --t_end {t_end} --max-elapsed {max_elapsed} "
        print(f"run: {command} \n")

        cluster.submit_command(
            command,
            name_run=name_run,
            nb_nodes=nb_nodes,
            walltime=walltime,
            nb_mpi_processes=nb_mpi_processes,
            omp_num_threads=1,
            delay_signal_walltime=300,
            ask=False,
            dependency="singleton",
        )

    else:
        print(
            f"More than one simulation with "
            f"{params:40s} \t"
            f"Nothing is done"
        )