"""
submit_bench_fluidsim.py
===============

Script to run the benchmarks of fluidsim (Script to run the benchmarks of fluidsim (https://fluidsim.readthedocs.io/en/latest/test_bench_profile.html)
Once you have runned many runs, you can run fluidsim-bench-analysis

Exemple:
python submit_bench_fluidsim.py
cd $WORK/fluidsim_bench
fluidsim-bench-analysis 640 1280 1280 -i . -s ns3d.strat

"""

from fluidjean_zay import cluster, JeanZay

# Usefull to solve the problem of long import # TODO: Remove it when it is solved.
cluster.commands_setting_env.append(
    "export TRANSONIC_MPI_TIMEOUT=100"
)


def submit(nb_nodes):
    nb_cores_per_node = cluster.nb_cores_per_node
    nb_mpi = nb_procs = nb_nodes * nb_cores_per_node

    cluster.submit_command(
        'fluidsim-bench -s ns3d.strat 640 1280 1280 '
        '-o $WORK/fluidsim_bench '
        '-t "fft3d.mpi_with_pfft" '
        '-it 20',
        name_run='fluidsim-bench_640_1280_1280_{:02d}'.format(nb_mpi),
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        walltime='01:00:00',
        nb_mpi_processes=nb_mpi, omp_num_threads=1,
        ask=False,
        delay_signal_walltime=None)


for nb_nodes in [1, 2, 4, 8]:
    submit(nb_nodes)
