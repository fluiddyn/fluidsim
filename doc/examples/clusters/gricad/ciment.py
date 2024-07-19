from oar import ClusterOAR


class ClusterOARGuix(ClusterOAR):
    resource_conditions: str
    use_oar_envsh: bool

    def check_oar(self):
        pass

    def _parse_cores_procs(self, nb_nodes, nb_cores_per_node, nb_mpi_processes):
        """Parse number of cores per node and MPI processes when these are
        None.

        """
        if not isinstance(nb_nodes, int) and nb_nodes > 0:
            raise ValueError("nb_nodes has to be a positive integer")

        if nb_cores_per_node is None:
            if nb_mpi_processes is not None and isinstance(nb_mpi_processes, int):
                nb_cores_per_node = nb_mpi_processes // nb_nodes
            else:
                nb_cores_per_node = self.nb_cores_per_node
        elif nb_cores_per_node > self.nb_cores_per_node:
            raise ValueError("Too many cores...")

        if nb_mpi_processes == "auto":
            nb_mpi_processes = nb_cores_per_node * nb_nodes

        return nb_cores_per_node, nb_mpi_processes


class DahuGuix(ClusterOARGuix):

    name_cluster = "dahu"
    has_to_add_name_cluster = False
    frontends = ["dahu", "dahu-oar3"]
    use_oar_envsh = False

    after_exec = (
        "~/.config/guix/current/bin/guix shell -E ^OMPI -E ^OAR -E ^OMP \\\n"
        "  -m manifest.scm -f python-fluidsim.scm \\\n  -- "
    )

    commands_setting_env = [
        "source /applis/site/guix-start.sh",
        "export OMPI_MCA_plm_rsh_agent=/usr/bin/oarsh",
        "export OMPI_MCA_btl_openib_allow_ib=true",
        "export OMPI_MCA_pml=cm",
        "export OMPI_MCA_mtl=psm2",
        '''MPI_PREFIX="`guix shell -m manifest.scm -f python-fluidsim.scm -- /bin/sh -c 'echo $GUIX_ENVIRONMENT'`"''',
    ]


class DahuGuixDevel(DahuGuix):
    devel = True
    frontends = ["dahu-oar3"]



class DahuGuix16_6130(DahuGuix):
    nb_cores_per_node = 16
    resource_conditions = "cpumodel='Gold 6130' and n_cores=16"


class DahuGuix32_6130(DahuGuix):
    nb_cores_per_node = 32
    resource_conditions = "cpumodel='Gold 6130' and n_cores=32"


class DahuGuix24_6126(DahuGuix):
    nb_cores_per_node = 24
    resource_conditions = "cpumodel='Gold 6126' and n_cores=24"


class DahuGuix32_5218(DahuGuix):
    nb_cores_per_node = 32
    resource_conditions = "cpumodel='Gold 5218' and n_cores=32"


class DahuGuix16_6244(DahuGuix):
    nb_cores_per_node = 16
    resource_conditions = "cpumodel='Gold 6244' and n_cores=16"
