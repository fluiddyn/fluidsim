"""
submit_bench_fluidfft.py
===============

Script to run the benchmarks of fluidfft (https://fluidfft.readthedocs.io/en/latest/bench.html)
Once you have runned many runs, you can run fluidfft-bench-analysis

Exemple:
python submit_bench_fluidfft.py
cd $WORK/fluidfft_bench
fluidfft-bench-analysis 128 128 128 -i .

"""

from fluiddyn.clusters.idris import JeanZay as Cluster

cluster = Cluster()

cluster.commands_setting_env += [
    "conda activate env_fluidsim",  # TODO: maybe this line should go into fluiddyn
]


def submit(nb_nodes, nb_cores_per_node=None):
    if nb_cores_per_node is None:
        nb_cores_per_node = cluster.nb_cores_per_node
    nb_mpi = nb_cores_per_node*nb_nodes
    cluster.submit_command(
        'fluidfft-bench 512 512 512 '
        '-o $WORK/fluidfft_bench '
        '-n 20',
        name_run='fluidfft-bench_{:02d}'.format(nb_mpi),
        nb_nodes=nb_nodes,
        # nb_cores_per_node=nb_cores_per_node,
        nb_cores_per_node=cluster.nb_cores_per_node,
        walltime='00:40:00',
        nb_mpi_processes=nb_mpi, omp_num_threads=1,
        ask=False,
        delay_signal_walltime=None)


nb_nodes = 1
for nb_cores_per_node in [2, 4, 8, 10, 12, 16, 20]:
    if nb_cores_per_node > cluster.nb_cores_per_node:
        continue
    submit(nb_nodes, nb_cores_per_node)

for nb_nodes in [2, 4]:
    submit(nb_nodes)


