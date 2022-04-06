"""
Needs some memory:

```
oarsub -I -l "{cluster='calcul2'}/nodes=1/core=4"
conda activate env_fluidsim
python submit640.py
```

"""

import subprocess
from time import sleep
from math import pi

from fluiddyn.clusters.oar import get_job_id, get_job_info
from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from util import cluster, path_base, couples640

nh = 640

if nh == 640:
    t_end = 30.0
    t_init = 20.0
else:
    raise NotImplementedError

nh_init = nh // 2

paths = sorted(path_base.glob(f"aniso/ns3d.strat*_{nh}x{nh}*"))
paths_init = sorted(path_base.glob(f"aniso/ns3d.strat*_{nh_init}x{nh_init}*"))


def filter_path(paths, Rb, N):
    return [
        p for p in paths if f"_Rb{Rb:.3g}_" in p.name and f"_N{N}_" in p.name
    ][0]


for N, Rb in sorted(couples640):

    name_1st_run = f"from_modified_resol_nx{nh}_Rb{Rb}_N{N}"
    job_id = get_job_id(name_1st_run)
    try:
        path = filter_path(paths, Rb, N)
    except IndexError:
        if job_id is None:

            try:
                path_init = filter_path(paths_init, Rb, N)
            except IndexError:
                print(
                    f"Cannot do anything for nx{nh}_Rb{Rb}_N{N} because no init directory"
                )
                continue

            t_start, t_last = times_start_last_from_path(path_init)
            if t_last < t_init:
                try:
                    estimated_remaining_duration = (
                        get_last_estimated_remaining_duration(path_init)
                    )
                except RuntimeError:
                    estimated_remaining_duration = "?"

                print(
                    f"Cannot launch {name_1st_run} because the coarse "
                    "simulation is not finished\n"
                    f"  ({t_last=} < {t_init=}, {estimated_remaining_duration = })"
                )
                continue

            try:
                path_init_file = next(
                    path_init.glob(f"State_phys_{nh}x{nh}*/state_phys_t*.h5")
                )
            except StopIteration:
                subprocess.run(
                    ["fluidsim-modif-resolution", str(path_init), "2"],
                    check=True,
                )
                path_init_file = next(
                    path_init.glob(f"State_phys_{nh}x{nh}*/state_phys_t*.h5")
                )

            period_spatiotemp = min(2 * pi / (N * 8), 0.03)

            command = (
                f"fluidsim-restart {path_init_file} --t_end {t_end} --new-dir-results "
                "--modify-params 'params.nu_4 /= 10; params.output.periods_save.phys_fields = 0.5; "
                "params.output.periods_print.print_stdout = 0.02; "
                f"params.output.periods_save.spatiotemporal_spectra = {period_spatiotemp}'"
            )

            cluster.submit_command(
                command,
                name_run=name_1st_run,
                nb_nodes=1,
                walltime="04:00:00",
                nb_mpi_processes=20,
                omp_num_threads=1,
                delay_signal_walltime=300,
                ask=False,
            )

            while job_id is None:
                job_id = get_job_id(name_1st_run)
                sleep(0.2)
        else:
            print(
                f"Nothing to do for nx{nh}_Rb{Rb}_N{N} because first job is "
                "already launched and the simulation directory is not created"
            )
            continue

    else:
        command = f"fluidsim-restart {path}"
        name_run = command.split()[0] + f"_nx{nh}_Rb{Rb}_N{N}"

        t_start, t_last = times_start_last_from_path(path)
        if t_last >= t_end:
            params = f"{N=} {Rb=} {nh=}"
            print(f"{params:40s}: completed")
            continue

        try:
            estimated_remaining_duration = get_last_estimated_remaining_duration(
                path
            )
        except RuntimeError:
            estimated_remaining_duration = "?"

        print(f"{path.name}: {t_last = }, {estimated_remaining_duration = }")

        if get_job_id(name_run) is not None:
            info = get_job_info(name_run)
            print(
                "  Nothing to do because the idempotent job is already launched "
                f"(job {info['id']}, {info['status']}, {info['duration']} / {info['walltime']})"
            )
            continue

        cluster.submit_command(
            command,
            name_run=name_run,
            nb_nodes=1,
            walltime="12:00:00",
            nb_mpi_processes=20,
            omp_num_threads=1,
            delay_signal_walltime=300,
            ask=False,
            idempotent=True,
            anterior=job_id,
        )
