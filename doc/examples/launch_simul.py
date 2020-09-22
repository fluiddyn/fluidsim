from fluiddyn.clusters.legi import Calcul as Cluster

# or
# from fluiddyn.clusters.legi import Calcul7 as Cluster
# or
# from fluiddyn.clusters.legi import Calcul8 as Cluster

cluster = Cluster()

cluster.commands_setting_env = [
    "source /etc/profile",
    'export PATH="$HOME/miniconda3/bin:$PATH"',
]

cluster.submit_script(
    "simul_ns2d.py",
    name_run="fld_example",
    nb_cores_per_node=4,
    nb_mpi_processes=4,
    omp_num_threads=1,
)
