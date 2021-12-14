import sys

from fluiddyn.clusters.legi import Calcul2 as Cluster

cluster = Cluster()

if "infiniband" in sys.argv:
    infiniband = True
else:
    infiniband = False

print(f"{infiniband = }")

cluster.commands_setting_env = [
    "source /etc/profile",
]

if infiniband:
    cluster.commands_setting_env.extend(["module load openmpi/4.0.5-ib"])
    resource_conditions = "net='ib' and os='buster'"
    name_run = "bench_cpp_mpi_ib"
    command = "./bench_ib.out"
else:
    cluster.commands_setting_env.extend(
        [
            "conda activate env_fluidsim_no_ib",
        ]
    )
    name_run = "bench_cpp_mpi_no_ib"
    resource_conditions = "os='stretch'"
    command = "./bench_no_ib.out"

resource_conditions = "net='ib' and os='buster'"

nb_nodes = 2

cluster.submit_command(
    command,
    name_run=name_run,
    nb_nodes=nb_nodes,
    nb_cores_per_node=1,
    nb_mpi_processes=nb_nodes,
    omp_num_threads=1,
    walltime="0:10:00",
    ask=False,
    resource_conditions=resource_conditions,
)
