from time import sleep

from fluiddyn.clusters.oar import get_job_id
from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from util import cluster, path_base, get_ratio_nh_nz, couples320

nh = 320
t_end = 20.0

paths = sorted(path_base.glob(f"aniso/ns3d.strat*_{nh}x{nh}*"))
walltime = "04:00:00"

for N, Rb in sorted(couples320):

    ratio_nh_nz = get_ratio_nh_nz(N)
    nz = nh // ratio_nh_nz

    nb_mpi_processes = int(round(nz / 2))
    assert nb_mpi_processes == nz / 2
    nb_mpi_processes = min(20, nb_mpi_processes)

    name_1st_run = f"run_simul_toro_nx{nh}_Rb{Rb}_N{N}"
    job_id = get_job_id(name_1st_run)
    try:
        path = [
            p for p in paths if f"_Rb{Rb:.3g}_" in p.name and f"_N{N}_" in p.name
        ][0]
    except IndexError:
        if job_id is None:
            command = f"./run_simul_toro.py -R {Rb} -N {N} --ratio-nh-nz {ratio_nh_nz} -nz {nz} --t_end {t_end}"

            cluster.submit_command(
                command,
                name_run=name_1st_run,
                nb_nodes=1,
                walltime=walltime,
                nb_mpi_processes=nb_mpi_processes,
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

        command = f"fluidsim-restart {path}"
        name_run = command.split()[0] + f"_nx{nh}_Rb{Rb}_N{N}"

        if get_job_id(name_run) is not None:
            print(f"Nothing to do because the idempotent job is already launched")
            continue

        cluster.submit_command(
            command,
            name_run=name_run,
            nb_nodes=1,
            walltime=walltime,
            nb_mpi_processes=nb_mpi_processes,
            omp_num_threads=1,
            delay_signal_walltime=300,
            ask=False,
            idempotent=True,
            anterior=job_id,
        )
