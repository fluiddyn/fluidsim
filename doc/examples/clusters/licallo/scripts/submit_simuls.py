from fluidlicallo import cluster
from math import pi

Ns = [20, 50, 100]
Rbs = [1, 5, 10, 20, 40]
nu_2 = 0.0
coef_nu4 = 1.5  # pi ** (10 / 3)
delta_angle = 0.2
ratio_nh_nz = 4
projs = ["None", "poloidal"]
# projs = ["poloidal"]
nz = 160
t_end = 50.0

nb_nodes = 2
nb_cores_per_node = cluster.nb_cores_per_node
nb_procs = nb_mpi_processes = nb_nodes * nb_cores_per_node

walltime = "23:55:00"
max_elapsed = "23:45:00"
type_fft = "'fluidfft.fft3d.mpi_with_fftwmpi3d'"


for N in Ns:
    Fh = 1.0 / N
    for proj in projs:
        command = (
            f"./run_simul.py -N {N} -nu {nu_2} -nz {nz} --coef-nu4 {coef_nu4} --ratio-nh-nz {ratio_nh_nz} "
            f"--spatiotemporal-spectra --t_end {t_end} --max-elapsed {max_elapsed} "
            f'--modify-params "'
            f"params.oper.type_fft = {type_fft}; "
            f"params.forcing.tcrandom_anisotropic.delta_angle = {delta_angle}; "
            f'" '
        )
        if proj == "poloidal" or "toroidal":
            command.append(f"--projection={proj}")
        else:
            print(
                'Projection (variable proj) must be "None", "poloidal", or "toroidal"'
            )

        cluster.submit_command(
            f"{command}",
            name_run=f"ns3d.strat_proj{proj}_Fh{Fh:.3e}_ratio{ratio_nh_nz}",
            nb_nodes=nb_nodes,
            nb_cores_per_node=nb_cores_per_node,
            nb_mpi_processes=nb_mpi_processes,
            omp_num_threads=1,
            ask=True,
            walltime=walltime,
        )
