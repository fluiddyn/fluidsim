"""
submit_bench_fluidfft.py
===============

Script to run the benchmarks of fluidfft (https://fluidfft.readthedocs.io/en/latest/bench.html)
Once you have runned many runs, you can run fluidfft-bench-analysis

Exemple:
python submit_bench_fluidfft.py
cd $WORK/fluidfft_bench
fluidfft-bench-analysis 320 320 320 -i .

"""

from fluiddyn.clusters.idris import JeanZay as Cluster

cluster = Cluster()

cluster.commands_setting_env += [
    "conda activate env_fluidsim",  # TODO: maybe this line should go into fluiddyn
]


def submit(nb_nodes):
    nb_cores_per_node = cluster.nb_cores_per_node
    nb_mpi = nb_procs = nb_nodes * nb_cores_per_node

    cluster.submit_command(
        'fluidsim-bench -s ns3d.strat 320 320 320 '
        '-o $WORK/fluidfft_bench '
        '-n 20',
        name_run='fluidfft-bench_{:02d}'.format(nb_mpi),
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        walltime='00:40:00',
        nb_mpi_processes=nb_mpi, omp_num_threads=1,
        ask=False,
        delay_signal_walltime=None)


for nb_nodes in [1, 2, 3, 4]:
    submit(nb_nodes)


