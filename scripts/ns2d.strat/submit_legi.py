import argparse
from fluiddyn.clusters.legi import Calcul

# To launch script (for gamma = 0.2):
# python submit_legi.py 0.2

parser = argparse.ArgumentParser()
parser.add_argument("gamma", type=float)
args = parser.parse_args()

cluster = Calcul()
cluster.commands_setting_env.append(
    'export FLUIDSIM_PATH="/fsnet/project/meige/2015/15DELDUCA/DataSim"'
)

name_run_root = f"find_coeff_nu8_gamma={args.gamma}"

walltime = "24:00:00"
nb_proc = 8

command_to_submit = f"python coeff_diss.py {args.gamma}"

cluster.submit_command(
    command_to_submit,
    name_run=name_run_root,
    nb_cores_per_node=nb_proc,
    walltime=walltime,
    nb_mpi_processes=nb_proc,
    omp_num_threads=1,
    idempotent=True,
    delay_signal_walltime=300,
    ask=False,
)
