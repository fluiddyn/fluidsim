"""Class for sim.turb_model (:mod:`fluidsim.base.turb_model.base`)
==================================================================

Provides:

.. autoclass:: TurbModel
   :members:
   :private-members:

"""

from fluidsim_core.params import iter_complete_params


def modif_infosolver_turb_model(info_solver):

    try:
        classes_TurbModel = getattr(info_solver.classes, "TurbModel")
    except AttributeError:
        classes_TurbModel = info_solver.classes._set_child(
            "TurbModel",
            attribs={
                "class_name": "TurbModel",
                "module_name": "fluidsim.base.turb_model.base",
            },
        )
        if hasattr(info_solver, "_cached_imported_classes"):
            info_solver._set_internal_attr("_cached_imported_classes", None)
    try:
        getattr(classes_TurbModel, "classes")
    except AttributeError:
        classes_TurbModel._set_child("classes")


class TurbModel:
    _name_task = "turb_model"

    @classmethod
    def _complete_params_with_default(cls, params, info_solver):
        """Complete the *params* container."""
        p_turb_model = params._set_child(
            cls._name_task,
            attribs={"enable": False, "type": ""},
        )

        p_turb_model._set_doc(
            """
        See :mod:`fluidsim.base.turb_model`.

        enable: bool

          Enable the use of a turbulent model.

        type : str

          Type of turb model.
        """
        )

        dict_classes = info_solver.classes.TurbModel.import_classes()
        iter_complete_params(params, info_solver, dict_classes)

    def __init__(self, sim):
        params = sim.params
        self.type_model = params.turb_model.type

        dict_classes = sim.info.solver.classes.TurbModel.import_classes()

        try:
            cls_model = dict_classes[self.type_model]
        except KeyError:
            raise ValueError(
                f"Wrong value ('{self.type_model}') of params.turb_model.type. "
                f"It should be in {dict_classes.keys()}"
            )

        self._model = cls_model(sim)

    def get_forcing(self, **kwargs):
        return self._model.get_forcing(**kwargs)
