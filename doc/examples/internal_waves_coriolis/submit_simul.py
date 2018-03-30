
from fluiddyn.clusters.legi import Calcul8 as Cluster
cluster = Cluster()

cluster.commands_setting_env = [
    'source /etc/profile',
    'export PATH="/home/users/augier3pi/opt/miniconda3/bin:$PATH"',
    'export FLUIDSIM_PATH=/fsnet/project/watu/2018/18INTERNAL/Data_simul'
]


cluster.submit_script(
    'simul_ns3dstrat_waves.py', name_run='fls_internal_waves',
    nb_cores_per_node=cluster.nb_cores_per_node,
    nb_mpi_processes=cluster.nb_cores_per_node, omp_num_threads=1)
