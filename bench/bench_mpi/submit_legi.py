from fluiddyn.clusters.legi import Calcul2 as Cluster

cluster = Cluster()

infiniband = True

cluster.commands_setting_env = [
    "source /etc/profile",
    "export FLUIDSIM_PATH=/fsnet/project/watu/2020/20MILESTONE",
    'export PATH="$HOME/miniconda3/bin:$PATH"',
    "source $HOME/init_conda.sh",
]

if infiniband:
    cluster.commands_setting_env.extend(
        ["conda activate env_fluidsim_ib", "module load openmpi/4.0.5-ib"]
    )
    resource_conditions = "net='ib' and os='buster'"
    name_run = "bench_mpi_ib"
else:
    cluster.commands_setting_env.extend(
        [
            "conda activate env_fluidsim_no_ib",
        ]
    )
    name_run = "bench_mpi_no_ib"
    resource_conditions = "os='stretch'"

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
