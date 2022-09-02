import os
from pathlib import Path
import subprocess
from util import path_base_jeanzay

from fluidjean_zay import cluster

nh = 640
coef_change_reso = 2
new_nh = nh * coef_change_reso

path_init = path_base_jeanzay

paths = sorted(path_init.glob(f"ns3d.strat_polo*_{nh}x{nh}*"))

for path in paths:
    try:
        new_path = next(
            path.glob("State_phys_{new_nh}x{new_nh}*/state_phys_*.h5")
        )
    except StopIteration:
        name_run = f"modif_reso_polo_nx{nh}"
        command = f"srun fluidsim-modif-resolution {path} {coef_change_reso}"
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
            walltime="00:15:00",
            project="uzc@cpu",
            partition="prepost",
        )

    else:
        print(f"{new_path} already created")
