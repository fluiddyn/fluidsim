from fluidjean_zay import cluster

out_dir = "/gpfswork/rech/zmr/uey73qw/bench"

# nh = 1280
# nbs_nodes = [2, 4, 8, 16]

nh = 2560
nbs_nodes = [4, 8, 16, 32, 64]

aspect_ratios = [2, 4, 8, 16]

nb_cores_per_node = cluster.nb_cores_per_node

for aspect_ratio in aspect_ratios:
    nz = nh // aspect_ratio

    for nb_nodes in nbs_nodes:
        nb_mpi_processes = nb_cores_per_node * nb_nodes

        cluster.submit_command(
            f"fluidfft-bench {nz} {nh} {nh} -o {out_dir}",
            name_run=f"fluidfft-bench_{nz}x{nh}x{nh}_np{nb_mpi_processes}",
            nb_nodes=nb_nodes,
            nb_cores_per_node=nb_cores_per_node,
            nb_mpi_processes=nb_mpi_processes,
            omp_num_threads=1,
            ask=False,
            walltime="02:00:00",
        )

        cluster.submit_command(
            f"fluidsim-bench -s ns3d.strat {nz} {nh} {nh} -o {out_dir} "
            '-t "fft3d.mpi_with_p3dfft" '
            "-it 20",
            name_run=f"fluidfft-bench_{nz}x{nh}x{nh}_np{nb_mpi_processes}",
            nb_nodes=nb_nodes,
            nb_cores_per_node=nb_cores_per_node,
            walltime="02:00:00",
            nb_mpi_processes=nb_mpi_processes,
            omp_num_threads=1,
            ask=False,
            delay_signal_walltime=None,
        )
