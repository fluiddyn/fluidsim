# python submit_bug_leak_memory.py

from fluiddyn.clusters.legi import Calcul7

cluster = Calcul7()
nb_proc = cluster.nb_cores_per_node // 2

R = 1
F = 0.5
factor_diss = 4

command_to_submit = 'python bug_leak_memory.py {} {} {}'.format(
    R, F, factor_diss)

cluster.submit_command(
    command_to_submit,
    name_run='fluidsim_bug_leak_memory{}'.format(factor_diss),
    nb_cores_per_node=cluster.nb_cores_per_node,
    walltime='6:00:00',
    nb_mpi_processes=nb_proc,
    ask=False)
