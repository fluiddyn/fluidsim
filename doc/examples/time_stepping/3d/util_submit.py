from fluiddyn.clusters.legi import Calcul8 as Cluster

max_elapsed = "07:40:00"
walltime = "08:00:00"

cluster = Cluster()

cluster.commands_setting_env = [
    "source /etc/profile",
    'export PATH="$HOME/miniconda3/bin:$PATH"'
    "export FLUIDSIM_PATH=/fsnet/project/watu/2020/20MILESTONE",
]

nb_nodes = 1


def submit_simul(coef_dealiasing, nx, type_time_scheme, cfl_coef=None):

    if nx < 480:
        nb_cores_per_node = 10
    else:
        nb_cores_per_node = 20

    nb_mpi_processes = nb_nodes * nb_cores_per_node

    command = (
        f"run_simul.py -cd {coef_dealiasing} -nx {nx} --type_time_scheme {type_time_scheme} "
        f'--max-elapsed "{max_elapsed}"'
    )

    if cfl_coef:
        command += f" -cfl {cfl_coef}"

    name_run = f"{type_time_scheme}_trunc{coef_dealiasing:.3f}"

    if cfl_coef:
        name_run += f"_cfl{cfl_coef}"

    print(f"submitting:\npython {command}")

    if not cluster:
        return

    try:
        cluster.submit_script(
            command,
            name_run=name_run,
            nb_nodes=nb_nodes,
            nb_cores_per_node=nb_cores_per_node,
            nb_mpi_processes=nb_mpi_processes,
            omp_num_threads=1,
            idempotent=True,
            ask=False,
            walltime=walltime,
        )
    except OSError:
        pass


if __name__ == "__main__":
    submit_simul(0.94, 128, "RK2_phaseshift", cfl_coef=0.2)
