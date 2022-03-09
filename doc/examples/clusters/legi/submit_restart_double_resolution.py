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
        "fluidsim-restart . --new-dir-results --add-to-t_end 0.5 "
        "--modify-params 'params.nu_4 /= 10; params.output.periods_save.phys_fields = 0.1'"
    ),
    name_run="fluiddyn",
    nb_nodes=1,
    walltime="24:00:00",
    nb_mpi_processes=cluster.nb_cores_per_node,
    omp_num_threads=1,
    delay_signal_walltime=300,
    ask=True,
    submit=True,
)
