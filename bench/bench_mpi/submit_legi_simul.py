import sys
from pathlib import Path

path_script = (
    Path(__file__).absolute().parent.parent.parent
    / "doc/examples/milestone/run_simul.py"
)

assert path_script.exists()

from fluiddyn.clusters.legi import Calcul2 as Cluster

cluster = Cluster()

if "no_infiniband" in sys.argv:
    infiniband = False
else:
    infiniband = True

print(f"{infiniband = }")

cluster.commands_setting_env = [
    "source /etc/profile",
    "export FLUIDSIM_PATH=/fsnet/project/watu/2020/20MILESTONE",
    "source /home/users/augier3pi/envs/mpi_ib/bin/activate",
]


if infiniband:
    name_run = "bench_sim_ib"
    cluster.commands_setting_env.extend(
        [
            "module load openmpi/4.0.5-ib",
            "export MPI4PY_RC_THREAD_LEVEL=serialized",
        ]
    )
else:
    name_run = "bench_sim_noib"

resource_conditions = "net='ib' and os='buster'"
nb_nodes = 2

cluster.submit_script(
    f"{path_script} -nc 5 -nypc 120 -D 0.5 -s 0.1",
    name_run=name_run,
    nb_nodes=nb_nodes,
    nb_cores_per_node=cluster.nb_cores_per_node,
    nb_mpi_processes=nb_nodes * cluster.nb_cores_per_node,
    omp_num_threads=1,
    walltime="0:30:00",
    ask=False,
    resource_conditions=resource_conditions,
)
