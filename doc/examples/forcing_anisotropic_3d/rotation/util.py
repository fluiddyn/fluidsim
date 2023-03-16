from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from pathlib import Path
import subprocess
import os

from fluidjean_zay import cluster

path_base_licallo = Path("/scratch/vlabarre/aniso_rotation/")
path_output_papermill_licallo = Path("/scratch/vlabarre/aniso_rotation/aniso_results_papermill")

path_base_azzurra = Path("/workspace/vlabarre/aniso_rotation/")
path_output_papermill_azzurra = Path(
    "/workspace/vlabarre/aniso_rotation/results_papermill"
)

path_base_jeanzay = Path("/gpfsscratch/rech/uzc/uey73qw/aniso_rotation/")
path_output_papermill_jeanzay = Path(
    "/gpfsscratch/rech/uzc/uey73qw/aniso_rotation/results_papermill"
)

path_base = path_base_jeanzay
path_output_papermill = path_output_papermill_jeanzay

coef_nu = 2.0
n_target = [320, 640, 1280]
Ro_target = [1.0, 10**(-0.5), 10**(-1), 10**(-1.5), 10**(-2), 10**(-2.5)]
walltime = "19:59:59"

def list_paths(Ro, n, NO_GEOSTROPHIC_MODES=False):
    # Find the paths of the simulations
    paths = sorted(path_base.glob(f"ns3d_polo*_{n}x{n}x{n}*"))
    print(paths)
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
    if n == 320:
        return "fftw3d"
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
        return "19:40:00" 
    if n == 640:
        return "19:00:00"
    if n == 1280:
        return "18:30:00"

def get_t_end(n):
    "Get end time of the simulation with resolution n"
    if n == 160:
        return 0
    elif n == 320:
        return 20.0
    elif n == 640:
        return 25.0
    elif n == 1280:
        return 30.0
    else:
        raise NotImplementedError

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
    n_lower = n // 2
    t_end_lower = get_t_end(n_lower)
    type_fft = type_fft_from_n(n)

    params = f"{Ro=} {n=} {NO_GEOSTROPHIC_MODES=}"
    
    name_run = f"run_simul_polo_Ro{Ro}_n{n}_NO_GEOSTROPHIC_MODES{NO_GEOSTROPHIC_MODES}"
    path_runs = list_paths(Ro, n, NO_GEOSTROPHIC_MODES=False)
    path_runs_lower = list_paths(Ro, n_lower, NO_GEOSTROPHIC_MODES=False)

    if is_job_submitted(name_run):
        print(
            f"Nothing to do for {params} because first job is "
            "already launched"
        )
        return


    if len(path_runs) == 0:
        if n == 320:
            command = (
                f"./run_simul_polo.py --Ro {Ro} -n {n} -coef_nu {coef_nu} --t_end {t_end} "
                f"--max-elapsed {max_elapsed} "
                f"--modify-params '"
                f'params.oper.type_fft = "fft3d.mpi_with_{type_fft}"; '
                f"'"
            )
            if NO_GEOSTROPHIC_MODES:
                command += f"--NO_GEOSTROPHIC_MODES {NO_GEOSTROPHIC_MODES}"
           
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
                t_start, t_last = times_start_last_from_path(path_runs_lower[0])
                if t_last < t_end_lower:
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
                        f"  ({t_last=} < {t_end_lower=}, {estimated_remaining_duration = })"
                    )
                    return

                
                path_state_lower = next(
                    path_runs_lower[0].glob(
                        f"State_phys_{n}x{n}x{n}*/state_phys_t*.h5"
                    )
                )

                if len(path_state_lower) == 0: 
                    modif_reso(path=path_runs_lower[0], n=n)
                elif len(path_state_lower) == 1: 
                    coef_change_reso = n / n_lower
                    coef_reduce_nu = coef_change_reso
                    command = (
                        f"fluidsim-restart {path_runs_lower} --t_end {t_end} --new-dir-results "
                        f"--max-elapsed {max_elapsed} "                 
                        f"--modify-params '"
                        f"params.nu_2 /= {coef_reduce_nu}; "
                        f'params.oper.type_fft = "fft3d.mpi_with_{type_fft}"; '
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
                    f"More than one state files in {path_runs_lower[0]}"
                    return
                  
            else:
                print(
                    f"Zero or more than one init directory with {params})",
                    f"Nothing is done",
                )
                return

    elif len(path_runs) == 1:
        t_start, t_last = times_start_last_from_path(path_runs[0])
        if t_last >= t_end:
            print(f"{params:40s}: completed")
            return

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
            ask=True,
            dependency="singleton",
        )

    else:
        print(
            f"More than one simulation with "
            f"{params:40s} \t"
            f"Nothing is done"
        )


def modif_reso(path, n, coef_change_reso=2):
    name_run = f"modif_reso_polo_{path}"
    command = f"srun fluidsim-modif-resolution {path} {coef_change_reso}"
    print(f"run command: {command}\n")

    # On Licallo
    #os.system(command)

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

