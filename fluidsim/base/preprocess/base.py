"""Base preprocess (:mod:`fluidsim.base.preprocess.base`)
=========================================================

Provides:

.. autoclass:: PreprocessBase
   :members:
   :private-members:

"""
from fluidsim_core.params import iter_complete_params


class PreprocessBase:
    _tag = "preprocess"

    @staticmethod
    def _complete_info_solver(info_solver, classes=None):
        """Complete the ParamContainer info_solver."""
        classesXML = info_solver.classes.Preprocess._set_child("classes")

        if classes is not None:
            for cls in classes:
                classesXML._set_child(
                    cls.tag,
                    attribs={
                        "module_name": cls.__module__,
                        "class_name": cls.__name__,
                    },
                )

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container."""
        attribs = {
            "enable": False,
            "init_field_scale": "unity",
            "init_field_const": 1.0,
            "viscosity_type": "laplacian",
            "viscosity_scale": "enstrophy_forcing",
            "viscosity_const": 1.0,
            "forcing_scale": "unity",
            "forcing_const": 1.0,
        }

        params._set_child("preprocess", attribs=attribs)

        dict_classes = info_solver.classes.Preprocess.import_classes()
        iter_complete_params(params, info_solver, dict_classes.values())

    def __init__(self, sim):
        self.params = sim.params.preprocess
        self.sim = sim
        self.oper = sim.oper
        self.output = sim.output

    # dict_classes = sim.info.solver.classes.Preprocess.import_classes()

    def __call__(self):
        if self.params.enable:
            self.output.print_stdout(
                "Preprocessing initial fields, and other parameters."
            )
