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
    "module purge",
]

if infiniband:
    name_venv = "venv_mpi_ib"
else:
    name_venv = "venv_mpi_noib"

name_run = "bench_py_" + name_venv

cluster.commands_setting_env.append(f"source {name_venv}/bin/activate")

if infiniband:
    cluster.commands_setting_env.append(
        "module load env/ib4openmpi  # export OMPI_MCA_pml=ucx",
    )

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
    resource_conditions="net='ib' and os='bullseye'",
)
