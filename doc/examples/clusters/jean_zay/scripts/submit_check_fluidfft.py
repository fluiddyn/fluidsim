"""
submit_check_fluidfft.py
========================

"""

from fluidjean_zay import cluster

nb_proc = nb_cores = 2
walltime = "00:10:00"

libraries = ["fftw1d", "fftwmpi3d", "pfft", "p3dfft"]

for lib in libraries:

    cluster.submit_script(
        f"check_fluidfft.py fft3d.mpi_with_{lib}",
        name_run=f"check_fluidfft_{lib}",
        nb_cores_per_node=nb_cores,
        walltime=walltime,
        nb_mpi_processes=nb_proc,
        omp_num_threads=1,
        ask=True,
    )
