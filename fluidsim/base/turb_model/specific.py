from fluidsim.extend_simul import SimulExtender
from fluidsim.base.setofvariables import SetOfVariables


class SmagorinskyModel(SimulExtender):
    _module_name = "fluidsim.base.turb_model.specific"
    tag = "smagorinsky"

    @classmethod
    def get_modif_info_solver(cls):
        """Create a function to modify ``info_solver``.

        Note that this function is called when the object ``info_solver`` has
        not yet been created (and cannot yet be modified)! This is why one
        needs to create a function that will be called later to modify
        ``info_solver``.

        """

        def modif_info_solver(info_solver):

            from fluidsim.base.turb_model.base import modif_infosolver_turb_model

            modif_infosolver_turb_model(info_solver)

            info_solver.classes.TurbModel.classes._set_child(
                cls.tag,
                attribs={
                    "module_name": cls._module_name,
                    "class_name": cls.__name__,
                },
            )

        return modif_info_solver

    @classmethod
    def complete_params_with_default(cls, params):
        pass

    def __init__(self, sim):
        self.sim = sim

        self.forcing_fft = SetOfVariables(
            like=sim.state.state_spect, info="forcing_fft", value=0.0
        )

    def get_forcing(self, **kwargs):
        return self.forcing_fft
