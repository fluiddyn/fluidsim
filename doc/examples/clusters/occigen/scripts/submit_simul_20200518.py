import sys

from fluidoccigen import cluster, Occigen

only_one = False

nu = 1e-6
max_elapsed = "23:30:00"

Ns = [0.1]
diameters = [0.25, 0.5]
velocities = [0.1, 0.2, 0.5]

D_max = max(diameters)

# first simulations
ny_per_cylinder = 96
nb_nodes = 1

# bigger simulations
# ny_per_cylinder = 144
# nb_nodes = 2

nb_cores_per_node = Occigen.nb_cores_per_node
nb_mpi_processes = nb_nodes * nb_cores_per_node

assert not ny_per_cylinder % nb_mpi_processes


def submit_simul(N, diameter, speed):

    command = (
        f"run_simul.py -N {N} -D {diameter} -s {speed} "
        f"-nypc {ny_per_cylinder} "
        f"--max-elapsed {max_elapsed}"
    )

    print(f"submitting:\npython {command}")

    if not cluster:
        return

    cluster.submit_script(
        command,
        name_run=f"milestoneN{N}D{diameter}s{speed}",
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        nb_mpi_processes=nb_mpi_processes,
        omp_num_threads=1,
        ask=False,
        walltime="23:59:58",
    )


if only_one:
    submit_simul(Ns[0], diameters[0], velocities[0])
    sys.exit()

for N in Ns:
    for diameter in diameters:
        for speed in velocities:
            submit_simul(N, diameter, speed)

submit_simul(0.5, 0.25, 0.2)
