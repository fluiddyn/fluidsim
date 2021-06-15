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
    "export MPI4PY_RC_THREAD_LEVEL=serialized",
]

if infiniband:
    name_venv = "mpi_ib"
else:
    name_venv = "mpi_noib"

name_run = "bench_py_" + name_venv

cluster.commands_setting_env.append(
    f"source /home/users/augier3pi/envs/{name_venv}/bin/activate"
)

if infiniband:
    cluster.commands_setting_env.append("module load openmpi/4.0.5-ib")

resource_conditions = "net='ib' and os='buster'"

nb_nodes = 2


cluster.submit_script(
    f"bench_point2point.py",
    name_run=name_run,
    nb_nodes=nb_nodes,
    nb_cores_per_node=1,
    nb_mpi_processes=nb_nodes,
    omp_num_threads=1,
    walltime="0:10:00",
    ask=False,
    resource_conditions=resource_conditions,
)
