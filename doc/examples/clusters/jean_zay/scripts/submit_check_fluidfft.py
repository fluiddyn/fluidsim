"""
submit_check_fluidfft.py
========================

"""

from fluiddyn.clusters.idris import JeanZay as Cluster

cluster = Cluster()

nb_proc = nb_cores = 2
walltime = "00:10:00"

libraries = ["fftw1d", "fftwmpi3d", "pfft"] # TODO: add p3dfft when the librairy is implemented


cluster.commands_setting_env += [
    "conda activate env_fluidsim",  # TODO: maybe this line should go into fluiddyn
    "export FLUIDSIM_PATH=$WORK/Fluidsim_Data/check_fluidfft",
]


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
