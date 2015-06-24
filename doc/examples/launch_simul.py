
from fluiddyn.clusters.legi import Calcul3 as Cluster
cluster = Cluster()


cluster.submit_script(
    'simul_ns2d.py', name_run='fld_example',
    nb_cores_per_node=cluster.nb_cores_per_node)
