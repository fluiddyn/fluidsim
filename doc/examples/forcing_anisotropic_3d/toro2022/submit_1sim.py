from fluiddyn.clusters.legi import Calcul8 as C

cluster = C()

cluster.commands_setting_env = [
    "source /etc/profile",
    ". $HOME/miniconda3/etc/profile.d/conda.sh",
    "conda activate env_fluidsim",
    "export FLUIDSIM_PATH=/fsnet/project/meige/2022/22STRATURBANIS",
]

N = 40
R = 5

assert N in [10, 20, 40]
assert R in [5, 10, 20, 40, 80, 160]

nh = 320
if N == 40:
    ratio_nh_nz = 8
elif N == 20:
    ratio_nh_nz = 4
elif N == 10:
    ratio_nh_nz = 2
else:
    raise NotImplementedError

nz = nh // ratio_nh_nz

cluster.submit_command(
    f"./run_simul_toro.py -R {R} -N {N} --ratio-nh-nz {ratio_nh_nz} -nz {nz}",
    name_run="fluiddyn",
    nb_nodes=1,
    walltime="10:00:00",
    nb_mpi_processes=4,  # cluster.nb_cores_per_node // 2,
    omp_num_threads=1,
    delay_signal_walltime=300,
    ask=True,
)
