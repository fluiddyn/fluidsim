"""
submit_bench_fluidfft.py
===============

Script to run the benchmarks of fluidfft (https://fluidfft.readthedocs.io/en/latest/bench.html)
Once you have runned many runs, you can run fluidfft-bench-analysis

Exemple:
python submit_bench_fluidfft.py
cd $WORK/fluidfft_bench
fluidfft-bench-analysis 640 1280 1280 -i .

"""

from fluidjean_zay import cluster


def submit(nb_nodes):
    nb_cores_per_node = cluster.nb_cores_per_node
    nb_mpi = nb_procs = nb_nodes * nb_cores_per_node

    cluster.submit_command(
        "fluidfft-bench 640 1280 1280 -o $WORK/fluidfft_bench -n 20",
        name_run=f"fluidfft-bench_640_1280_1280_{nb_mpi:02d}",
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        walltime="00:40:00",
        nb_mpi_processes=nb_mpi,
        omp_num_threads=1,
        ask=False,
        delay_signal_walltime=None,
    )


for nb_nodes in [1, 2, 4, 8]:
    submit(nb_nodes)
