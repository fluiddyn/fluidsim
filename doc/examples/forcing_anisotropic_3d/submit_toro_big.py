"""
Needs some memory:

```
oarsub -I -l "{cluster='calcul2'}/nodes=1/core=4"
conda activate env_fluidsim
python submit_toro_big.py
```

"""

import subprocess
from pathlib import Path
from time import sleep

from fluiddyn.clusters.legi import Calcul8 as C
from fluiddyn.clusters.oar import get_job_id
from fluidsim.util import times_start_last_from_path

cluster = C()

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS")

cluster.commands_setting_env = [
    "source /etc/profile",
    ". $HOME/miniconda3/etc/profile.d/conda.sh",
    "conda activate env_fluidsim",
    f"export FLUIDSIM_PATH={path_base}",
]

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
    return [p for p in paths if f"_Rb{Rb}_" in p.name and f"_N{N}_" in p.name][0]


for N in [10, 20, 40]:
    for Rb in [5, 10, 20, 40, 80, 160]:
        if N == 40 and Rb == 160:
            continue

        if Rb == 5 and N == 10:
            continue

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
                    print(
                        f"Cannot do anything for nx{nh}_Rb{Rb}_N{N} because {t_last=} < {t_init=}"
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

                command = (
                    f"fluidsim-restart {path_init_file} --t_end {t_end} --new-dir-results "
                    "--modify-params 'params.nu_4 /= 10; params.output.periods_save.phys_fields = 0.5; "
                    "params.output.periods_save.spatiotemporal_spectra = 2 * pi / (params.N * 4)'"
                )

                cluster.submit_command(
                    command,
                    name_run=name_1st_run,
                    nb_nodes=1,
                    walltime="04:00:00",
                    nb_mpi_processes=10,
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

            t_start, t_last = times_start_last_from_path(path)
            if t_last > t_end:
                print(f"Nothing to do for {path.name} because {t_last=} > {t_end=}")
                continue
            print(f"{path.name}: {t_last = }")

            command = f"fluidsim-restart {path}"
            name_run = command.split()[0] + f"_nx{nh}_Rb{Rb}_N{N}"

            if get_job_id(name_run) is not None:
                print(
                    f"Nothing to do for {path.name} because the idempotent job is "
                    "already launched"
                )
                continue

            cluster.submit_command(
                command,
                name_run=name_run,
                nb_nodes=1,
                walltime="04:00:00",
                nb_mpi_processes=10,
                omp_num_threads=1,
                delay_signal_walltime=300,
                ask=False,
                idempotent=True,
                anterior=job_id,
            )
