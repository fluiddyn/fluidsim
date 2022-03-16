from pathlib import Path

from fluiddyn.clusters.legi import Calcul8 as C

cluster = C()

path_base = Path("/fsnet/project/meige/2022/22STRATURBANIS")

cluster.commands_setting_env = [
    "source /etc/profile",
    ". $HOME/miniconda3/etc/profile.d/conda.sh",
    "conda activate env_fluidsim",
    f"export FLUIDSIM_PATH={path_base}",
]

nh = 160

paths = sorted(path_base.glob(f"aniso/ns3d.strat*_{nh}x{nh}*"))


def get_ratio_nh_nz(N):
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

        try:
            path = [
                p for p in paths if f"_Rb{Rb}_" in p.name and f"_N{N}_" in p.name
            ][0]
        except IndexError:
            command = f"./run_simul_toro.py -R {Rb} -N {N} --ratio-nh-nz {ratio_nh_nz} -nz {nz}"
            idempotent = False
            walltime = "00:10:00"
        else:
            command = f"fluidsim-restart {path}"
            idempotent = True
            walltime = "01:00:00"

        name_run = command.split()[0]
        if name_run.startswith("./"):
            name_run = "run_simul_toro"
            
        cluster.submit_command(
            command,
            name_run=name_run,
            nb_nodes=1,
            walltime=walltime,
            nb_mpi_processes=4,
            omp_num_threads=1,
            delay_signal_walltime=300,
            idempotent=idempotent,
            ask=False,
        )
