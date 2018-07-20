from fluiddyn.clusters.legi import Calcul

cluster = Calcul()
cluster.commands_setting_env.append(
    'export FLUIDSIM_PATH="/fsnet/project/meige/2015/15DELDUCA/DataSim"')

name_run_root = "find_coeff_nu8"

walltime = '12:00:00'
nb_proc = 8

command_to_submit = "python coeff_diss.py"

cluster.submit_command(
    command_to_submit,
    name_run=name_run_root,
    nb_cores_per_node=nb_proc,
    walltime=walltime,
    nb_mpi_processes=nb_proc,
    omp_num_threads=1,
    idempotent=True, delay_signal_walltime=300, ask=False)
