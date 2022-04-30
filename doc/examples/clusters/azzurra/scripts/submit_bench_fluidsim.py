"""
submit_bench_fluidsim.py
===============

Script to run the benchmarks of fluidsim (Script to run the benchmarks of fluidsim (https://fluidsim.readthedocs.io/en/latest/test_bench_profile.html)
Once you have runned many runs, you can run fluidsim-bench-analysis

Exemple:
python submit_bench_fluidsim.py
cd /workspace/$USER/fluidsim_bench
fluidsim-bench-analysis 160 640 640 -i . -s ns3d.strat

"""

from fluidazzurra import cluster


def submit(nb_nodes):
    nb_cores_per_node = cluster.nb_cores_per_node
    nb_mpi_processes = nb_nodes * nb_cores_per_node

    cluster.submit_command(
        "fluidsim-bench -s ns3d.strat 160 640 640 "
        "-o /workspace/$USER/fluidsim_bench "
        '-t "fft3d.mpi_with_pfft" '
        "-it 20",
        name_run=f"fluidsim-bench_320_640_640_{nb_mpi_processes:02d}",
        account="turbulence",
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        walltime="02:00:00",
        nb_mpi_processes=nb_mpi_processes,
        omp_num_threads=1,
        ask=False,
        delay_signal_walltime=None,
    )


for nb_nodes in [1, 2, 4, 8]:
    submit(nb_nodes)
