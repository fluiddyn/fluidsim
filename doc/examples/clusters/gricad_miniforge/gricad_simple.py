from pathlib import Path

from fluiddyn.clusters.oar import ClusterOAR


path_miniforge = Path.home() / "miniforge3"


class Dahu(ClusterOAR):
    name_cluster = "dahu"
    has_to_add_name_cluster = False
    frontends = ["dahu", "dahu-oar3"]
    use_oar_envsh = False

    commands_setting_env = [
        "source /etc/profile",
        f"source {path_miniforge / 'etc/profile.d/conda.sh'}",
        "conda activate env-fluidsim-mpi",
    ]


class DahuDevel(Dahu):
    devel = True
    frontends = ["dahu-oar3"]


class Dahu16_6130(Dahu):
    nb_cores_per_node = 16
    resource_conditions = "cpumodel='Gold 6130' and n_cores=16"


class Dahu32_6130(Dahu):
    nb_cores_per_node = 32
    resource_conditions = "cpumodel='Gold 6130' and n_cores=32"


class Dahu24_6126(Dahu):
    nb_cores_per_node = 24
    resource_conditions = "cpumodel='Gold 6126' and n_cores=24"


class Dahu32_5218(Dahu):
    nb_cores_per_node = 32
    resource_conditions = "cpumodel='Gold 5218' and n_cores=32"


class Dahu16_6244(Dahu):
    nb_cores_per_node = 16
    resource_conditions = "cpumodel='Gold 6244' and n_cores=16"
