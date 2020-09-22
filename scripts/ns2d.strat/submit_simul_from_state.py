"""
submit_simul_from_state.py
==========================
"""
import argparse
from fluiddyn.clusters.legi import Calcul

parser = argparse.ArgumentParser()
parser.add_argument("gamma", type=float)
parser.add_argument("NO_SHEAR_MODES", type=int)
args = parser.parse_args()

cluster = Calcul()
cluster.commands_setting_env.append(
    'export FLUIDSIM_PATH="/fsnet/project/meige/2015/15DELDUCA/DataSim"'
)

name_run_root = "sim480_gamma={}_NO_SHEAR_MODES={}".format(
    args.gamma, bool(args.NO_SHEAR_MODES)
)

walltime = "24:00:00"
# nb_proc = cluster.nb_cores_per_node
nb_proc = 8

command_to_submit = "python simul_from_state.py {} {}".format(
    args.gamma, args.NO_SHEAR_MODES
)

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
