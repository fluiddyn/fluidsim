from fluiddyn.clusters.legi import Calcul8 as Cluster

cluster = Cluster()

cluster.commands_setting_env = [
    "source /etc/profile",
    'export PATH="$HOME/miniconda3/bin:$PATH"'
    "export FLUIDSIM_PATH=/fsnet/project/watu/2019/19INTSIM/sim_data",
]

cluster.submit_script(
    "simul_ns3dstrat_waves.py",
    name_run="fld_igw3d",
    nb_cores_per_node=10,
    nb_mpi_processes=10,
    omp_num_threads=1,
)
