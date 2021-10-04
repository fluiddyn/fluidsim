from fluiddyn.clusters.legi import Calcul8 as Cluster

cluster = Cluster()

cluster.commands_setting_env = [
    "source /etc/profile",
    ". $HOME/miniconda3/etc/profile.d/conda.sh",
    "conda activate env_fluidsim",
    "export FLUIDSIM_PATH=/fsnet/project/watu/2020/20MILESTONE",
]

# velocities = [0.05, 0.1, 0.2]
# diameters = [0.25, 0.5]

velocities = [0.1]
diameters = [0.5]

for diameter in diameters:
    for speed in velocities:
        cluster.submit_script(
            f"run_simul.py -D {diameter} -s {speed}",
            name_run="milestone",
            nb_cores_per_node=10,
            nb_mpi_processes=10,
            omp_num_threads=1,
            ask=False,
        )
