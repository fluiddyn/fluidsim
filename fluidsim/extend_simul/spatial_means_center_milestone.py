from . import SimulExtender


class SpatialMeansCenter(SimulExtender):
    _tag = "spatial_means_region"

    def __init__(self, output):
        self.output = output

    @classmethod
    def create_extended_Simul(cls, Simul):
        """Here we want to add a specific output"""

        def modif_info_solver(info_solver):

            info_solver.classes.Output.classes._set_child(
                "SpatialMeansCenter",
                attribs={
                    "module_name": "fluidsim.extend_simul.spatial_means_center_milestone",
                    "class_name": "SpatialMeansCenter",
                },
            )

        class NewInfoSolver(Simul.InfoSolver):
            pass

        cls.add_info_solver_modificator(NewInfoSolver, modif_info_solver)

        class NewSimul(Simul):
            InfoSolver = NewInfoSolver

        return NewSimul
