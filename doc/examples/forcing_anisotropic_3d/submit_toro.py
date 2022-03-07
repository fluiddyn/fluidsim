from fluiddyn.clusters.legi import Calcul8 as C

cluster = C()

cluster.commands_setting_env = [
    "source /etc/profile",
    ". $HOME/miniconda3/etc/profile.d/conda.sh",
    "conda activate env_fluidsim",
    "export FLUIDSIM_PATH=/fsnet/project/meige/2022/22STRATURBANIS",
]

cluster.submit_command(
    (
        "./run_simul_toro.py -nz 80 -N 40 -R 20"
    ),
    name_run="fluiddyn",
    nb_nodes=1,
    walltime="24:00:00",
    nb_mpi_processes=cluster.nb_cores_per_node,
    omp_num_threads=1,
    delay_signal_walltime=300,
    ask=True,
)
