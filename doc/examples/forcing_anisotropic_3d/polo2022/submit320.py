from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from fluidlicallo import cluster

import os

from util import (
    list_paths,
    get_ratio_nh_nz,
    couples320,
    is_job_submitted,
    get_t_end,
)

nh = 320

walltime = "23:59:59"
max_elapsed = "23:50:00"

for proj in ["None", "poloidal"]:
    for N, Rb in sorted(couples320):
        print("--------------------------------------------")
        ratio_nh_nz = get_ratio_nh_nz(N)
        t_end = get_t_end(N, nh)
        nz = nh // ratio_nh_nz

        nb_mpi_processes = int(round(nz / 2))
        assert nb_mpi_processes == nz / 2
        nb_mpi_processes = min(20, nb_mpi_processes)

        name_run = f"run_simul_polo_nx{nh}_Rb{Rb}_N{N}_proj{proj}"
        path_runs = list_paths(N=N, Rb=Rb, nh=nh, nz=nz, proj=proj)

        if is_job_submitted(name_run):
            print(
                f"Nothing to do for nx{nh}_Rb{Rb}_N{N}_proj{proj} because first job is "
                "already launched"
            )
            continue

        if len(path_runs) == 0:
            command = (
                f"./run_simul_polo.py -R {Rb} -N {N} --ratio-nh-nz {ratio_nh_nz} -nz {nz} --t_end {t_end} "
                f"--max-elapsed {max_elapsed} "
            )
            if proj == "poloidal":
                command += f" --projection {proj}"

            cluster.submit_command(
                command,
                name_run=name_run,
                nb_nodes=1,
                walltime=walltime,
                nb_mpi_processes=nb_mpi_processes,
                omp_num_threads=1,
                delay_signal_walltime=300,
                ask=False,
            )

        elif len(path_runs) == 1:
            t_start, t_last = times_start_last_from_path(path_runs[0])
            if t_last >= t_end:
                params = f"{proj=} {N=} {Rb=} {nh=}"
                print(f"{params:40s}: completed")
                continue

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
                nb_nodes=1,
                walltime=walltime,
                nb_mpi_processes=nb_mpi_processes,
                omp_num_threads=1,
                delay_signal_walltime=300,
                ask=False,
                dependency="singleton",
            )

        else:
            print(
                f"More than one simulation with {proj=}, {N=}, {Rb=}, {nh=}, {nz=})",
                f"Nothing is done",
            )
