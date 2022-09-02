"""
Needs some memory:

```
srun --pty --ntasks=1 --cpus-per-task=4 --hint=nomultithread --time=0:15:0 bash
conda activate env_fluidsim
python submit2560.py
```

"""

import subprocess
from math import pi
import os

from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from fluidjean_zay import cluster

from util import (
    list_paths_jeanzay,
    get_ratio_nh_nz,
    couples,
    is_job_submitted,
    get_t_end,
    type_fft_from_nh_nz,
    nb_nodes_from_nh_nz,
)

nh = 2560
nh_init = 1920

walltime = "19:59:59"
max_elapsed = "18:30:00"

nb_cores_per_node = cluster.nb_cores_per_node

for proj in ["None", "poloidal"]:
    for N, Rb in sorted(couples[nh]):
        print("--------------------------------------------")
        t_end = get_t_end(N, nh)
        t_init = get_t_end(N, nh_init)
        ratio_nh_nz = get_ratio_nh_nz(N)
        nz = nh // ratio_nh_nz
        type_fft = type_fft_from_nh_nz(nh, nz)
        nb_nodes = nb_nodes_from_nh_nz(nh, nz)
        nb_mpi_processes = nb_nodes * nb_cores_per_node

        name_run = f"run_simul_polo_nx{nh}_Rb{Rb}_N{N}_proj{proj}"
        path_runs = list_paths_jeanzay(N=N, Rb=Rb, nh=nh, nz=nz, proj=proj)

        if is_job_submitted(name_run):
            print(
                f"Nothing to do for nx{nh}_Rb{Rb}_N{N}_{proj} because the job is "
                "already launched"
            )
            continue

        if len(path_runs) == 0:
            # Case where we have to restart from a simulation with a coarse resolution
            if (N, Rb) in couples[nh_init]:
                path_inits = list_paths_jeanzay(
                    N=N, Rb=Rb, nh=nh_init, nz=nz, proj=proj
                )

                if len(path_inits) == 0:
                    print(
                        f"Cannot do anything for nx{nh}_Rb{Rb}_N{N}_proj{proj} because no init directory. Need to run a simulation at smaller resolution."
                    )
                    continue

                elif len(path_inits) == 1:
                    t_start, t_last = times_start_last_from_path(path_inits[0])
                    if t_last < t_init:
                        try:
                            estimated_remaining_duration = (
                                get_last_estimated_remaining_duration(
                                    path_inits[0]
                                )
                            )
                        except RuntimeError:
                            estimated_remaining_duration = "?"

                        print(
                            f"Cannot launch {name_run} because the coarse "
                            "simulation is not finished\n"
                            f"  ({t_last=} < {t_init=}, {estimated_remaining_duration = })"
                        )
                        continue

                    try:
                        path_init_file = next(
                            path_inits[0].glob(
                                f"State_phys_{nh}x{nh}*/state_phys_t*.h5"
                            )
                        )
                    except StopIteration:
                        name_run = f"modif_reso_polo_nx{nh}"
                        command = f"srun fluidsim-modif-resolution --t_approx {t_init} {path_inits[0]} 4/3"
                        print(f"run command: {command}\n")
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
                            walltime="01:30:00",
                            project="uzc@cpu",
                            partition="prepost",
                        )
                    else:
                        print(f"{path_init_file} already created")

                        period_spatiotemp = min(2 * pi / (N * 8), 0.03)
                        coef_decrease_nu4 = (nh / nh_init) ** (10 / 3)

                        command = (
                            f"fluidsim-restart {path_init_file} --t_end {t_end} --new-dir-results "
                            f"--max-elapsed {max_elapsed} "
                            f"--modify-params 'params.nu_4 /= {coef_decrease_nu4}; params.output.periods_save.phys_fields = 0.5; "
                            "params.output.periods_print.print_stdout = 0.02; "
                            f'params.oper.type_fft = "fft3d.mpi_with_{type_fft}"; '
                            f"params.output.periods_save.spatiotemporal_spectra = {period_spatiotemp}'"
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
                            project="uzc@cpu",
                        )

                else:
                    print(
                        f"More than one init directory with {N=}, {Rb=}, {nh_init=}, {nz=}, {proj=})",
                        f"Nothing is done",
                    )
                    continue

            else:
                print(f"({N}, {Rb}) not in couples{nh_init}\n")
                continue

        elif len(path_runs) == 1:
            # See if we restart the simulation
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
                f"{path_runs[0].name}: {t_last = }, {estimated_remaining_duration = }"
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
                project="uzc@cpu",
            )

        else:
            print(
                f"More than one simulation with {N=}, {Rb=}, {nh=}, {nz=}, {proj=})",
                f"Nothing is done",
            )
