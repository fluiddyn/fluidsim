from fluiddyn.clusters.legi import Calcul8 as Cluster

cluster = Cluster()

cluster.commands_setting_env = [
    "source /etc/profile",
    'export PATH="$HOME/miniconda3/bin:$PATH"'
    "export FLUIDSIM_PATH=/fsnet/project/watu/2019/19INTSIM/sim_data",
]

nz = 60

for amplitude in (0.02, 0.035, 0.05):
    cluster.submit_command(
        f"python simul_idempotent.py {amplitude} {nz} --max-elapsed 23:00:00",
        name_run="fld_igw3d",
        nb_cores_per_node=20,
        nb_mpi_processes=20,
        omp_num_threads=1,
        idempotent=True,
        walltime="24:00:00",
        ask=False,
    )
