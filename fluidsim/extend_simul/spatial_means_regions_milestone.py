"""Spatial means regions
========================

.. autoclass:: SpatialMeansRegions
   :members:
   :private-members:

"""

from . import SimulExtender


class SpatialMeansRegions(SimulExtender):
    """Specific output for the MILESTONE simulations

    It is still a work in progress.

    """
    _tag = "spatial_means_regions"
    _module_name = "fluidsim.extend_simul.spatial_means_regions_milestone"

    def __init__(self, output):
        self.output = output

    @classmethod
    def complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)
        params.output._set_child(cls._tag, attribs={"xmin": 2, "xmax": 4})

    @classmethod
    def create_extended_Simul(cls, Simul):
        """Here we want to add a specific output"""

        def modif_info_solver(info_solver):

            info_solver.classes.Output.classes._set_child(
                "SpatialMeansRegions",
                attribs={
                    "module_name": cls._module_name,
                    "class_name": cls.__name__,
                },
            )

        return super().create_extended_Simul(Simul, modif_info_solver)

    def _online_save(self):
        return NotImplemented

    def load(self):
        return NotImplemented

    def plot(self):
        return NotImplemented
