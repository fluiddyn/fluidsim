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

nh = 320
t_end = 20.0

paths = sorted(path_base.glob(f"aniso/ns3d.strat*_{nh}x{nh}*"))


def get_ratio_nh_nz(N):
    "Get the ratio nh/nz"
    if N == 40:
        return 8
    elif N == 20:
        return 4
    elif N == 10:
        return 2
    else:
        raise NotImplementedError


for N in [10, 20, 40]:
    for Rb in [5, 10, 20, 40, 80, 160]:
        if N == 40 and Rb == 160:
            continue

        ratio_nh_nz = get_ratio_nh_nz(N)
        nz = nh // ratio_nh_nz

        name_1st_run = f"run_simul_toro_nx{nh}_Rb{Rb}_N{N}"
        job_id = get_job_id(name_1st_run)
        try:
            path = [
                p for p in paths if f"_Rb{Rb}_" in p.name and f"_N{N}_" in p.name
            ][0]
        except IndexError:
            if job_id is None:
                command = f"./run_simul_toro.py -R {Rb} -N {N} --ratio-nh-nz {ratio_nh_nz} -nz {nz} --t_end {t_end}"
                walltime = "04:00:00"

                cluster.submit_command(
                    command,
                    name_run=name_1st_run,
                    nb_nodes=1,
                    walltime=walltime,
                    nb_mpi_processes=10,
                    omp_num_threads=1,
                    delay_signal_walltime=300,
                    ask=False,
                )

                while job_id is None:
                    job_id = get_job_id(name_1st_run)
                    sleep(1)
            else:
                print(
                    f"Nothing to do for nx{nh}_Rb{Rb}_N{N} because first job is "
                    "already launched and the simulation directory is not created"
                )
                continue

        else:

            t_start, t_last = times_start_last_from_path(path)
            if t_last > t_end:
                print(f"Nothing to do for {path.name} because t_last > t_end")
                continue

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