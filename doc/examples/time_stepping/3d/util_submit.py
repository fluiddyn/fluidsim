from fluiddyn.clusters.legi import Calcul8 as Cluster

max_elapsed = "07:40:00"
walltime = "08:00:00"

cluster = Cluster()

cluster.commands_setting_env = [
    "source /etc/profile",
    'export PATH="$HOME/miniconda3/bin:$PATH"'
    "export FLUIDSIM_PATH=/fsnet/project/meige/2020/20PHASESHIFT",
]


def submit_simul(
    coef_dealiasing,
    nx,
    type_time_scheme,
    cfl_coef=None,
    nb_proc=None,
    truncation_shape=None,
    Re=None,
):
    nb_nodes = 1
    if nb_proc is None:
        if nx < 480:
            nb_proc = 10
        if nx >= 640:
            nb_proc = 40
            nb_nodes = 2
        else:
            nb_proc = 20

    tmp = nx / 2 / nb_proc
    assert round(tmp) == tmp

    nb_mpi_processes = nb_proc

    command = (
        f"run_simul.py -cd {coef_dealiasing} -nx {nx} --type_time_scheme {type_time_scheme} "
        f'--max-elapsed "{max_elapsed}"'
    )

    name_run = f"{type_time_scheme}_trunc{coef_dealiasing:.3f}"

    if cfl_coef:
        command += f" -cfl {cfl_coef}"
        name_run += f"_cfl{cfl_coef}"

    if truncation_shape is not None:
        command += f" --truncation_shape {truncation_shape}"
        name_run += f"_{truncation_shape}"

    if Re is not None:
        command += f" --Re {Re}"
        name_run += f"_Re{Re}"

    print(f"submitting:\npython {command}")

    if not cluster:
        return

    try:
        cluster.submit_script(
            command,
            name_run=name_run,
            nb_nodes=nb_nodes,
            nb_cores_per_node=nb_proc // nb_nodes,
            nb_mpi_processes=nb_mpi_processes,
            omp_num_threads=1,
            idempotent=True,
            ask=False,
            walltime=walltime,
        )
    except OSError:
        pass


def submit_profile(
    coef_dealiasing,
    nx,
    type_time_scheme,
    t_end,
    cfl_coef=None,
    nb_pairs=1,
    nb_steps=None,
):

    if nx < 480:
        nb_cores_per_node = 10
    else:
        nb_cores_per_node = 20

    command = (
        f"run_profile.py -cd {coef_dealiasing} -nx {nx} "
        f"--type_time_scheme {type_time_scheme} --t_end {t_end} "
        f"--nb_pairs {nb_pairs}"
    )

    if cfl_coef:
        command += f" -cfl {cfl_coef}"

    if nb_steps:
        command += f" --nb_steps {nb_steps}"

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
            nb_nodes=1,
            nb_cores_per_node=nb_cores_per_node,
            nb_mpi_processes=1,
            omp_num_threads=1,
            idempotent=True,
            ask=False,
            walltime="02:00:00",
        )
    except OSError:
        pass


if __name__ == "__main__":
    submit_simul(0.94, 128, "RK2_phaseshift", cfl_coef=0.2)
