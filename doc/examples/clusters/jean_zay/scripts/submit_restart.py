from pathlib import Path

from fluidoccigen import cluster, Occigen

path = Path("/scratch/cnt0022/egi2153/augier/milestone/")

if not path.exists():
    path = Path("/data0/milestone_occigen")

paths = path.glob("ns3d.strat_960x288x96_V7.5x2.25x0.75_*_D0.25_2020-05-15_*")

nb_nodes = 1
nb_cores_per_node = Occigen.nb_cores_per_node
nb_mpi_processes = nb_nodes * nb_cores_per_node


for path in paths:

    command = f"run_simul_restart.py -p {path}"

    print(f"submitting:\npython {command}")

    title = "_".join(path.name.split("_")[3:-2])

    if not cluster:
        continue

    cluster.submit_script(
        command,
        name_run=f"restart_" + title,
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        nb_mpi_processes=nb_mpi_processes,
        omp_num_threads=1,
        ask=False,
        walltime="23:59:58",
    )
