"""
submit_bench_fluidsim.py
===============

Script to run the benchmarks of fluidsim (Script to run the benchmarks of fluidsim (https://fluidsim.readthedocs.io/en/latest/test_bench_profile.html)
Once you have runned many runs, you can run fluidsim-bench-analysis

Exemple:
python submit_bench_fluidsim.py
cd $WORK/fluidsim_bench
fluidsim-bench-analysis 1280 1280 320 -i . -s ns3d.strat

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
        '-o $WORK/fluidsim_bench '
        #'-t "all" ' # TODO: Does not work, I don't know why...
        '-it 4',
        name_run='fluidsim-bench_{:02d}'.format(nb_mpi),
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        walltime='02:00:00',
        nb_mpi_processes=nb_mpi, omp_num_threads=1,
        ask=False,
        delay_signal_walltime=None)


for nb_nodes in [1, 2, 3, 4]:
    submit(nb_nodes)
