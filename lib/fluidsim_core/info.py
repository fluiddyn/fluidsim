"""Info solver
==============

Container to encode the solver class hierarchy.

.. autofunction:: create_info_simul

.. autoclass:: InfoSolverCore
   :members:
   :private-members:

"""
from fluiddyn.util import import_class
from fluiddyn.util.paramcontainer import ParamContainer


def create_info_simul(info_solver, params):
    """Create a ParamContainer instance gathering info_solver and params."""
    info = ParamContainer(tag="info_simul")
    info._set_as_child(info_solver)
    info._set_as_child(params)
    return info


class InfoSolverCore(ParamContainer):
    """Contain the information on a solver."""

    def __init__(self, **kargs):

        if len(kargs) == 0 or ("path_file" in kargs and "tag" not in kargs):
            kargs["tag"] = "solver"

        super().__init__(**kargs)

        if (
            "tag" in kargs
            and kargs["tag"] == "solver"
            and "path_file" not in kargs
        ):
            self._init_root()

    def _init_root(self):

        self._set_attribs(
            {
                "module_name": "fluidsim_core.solver",
                "class_name": "SimulCore",
                "short_name": "Core",
            }
        )

        _ = self._set_child("classes")

    def import_classes(self):
        """Import the classes and return a dictionary."""
        if hasattr(self, "_cached_imported_classes"):
            return self._cached_imported_classes

        dict_classes = {}
        tags = self.classes._tag_children
        if len(tags) == 0:
            self._set_internal_attr("_cached_imported_classes", dict_classes)
            return dict_classes

        for tag in tags:
            cls = self.classes.__dict__[tag]
            try:
                module_name = cls.module_name
                class_name = cls.class_name
            except AttributeError:
                pass
            else:
                Class = import_class(module_name, class_name)
                dict_classes[cls._tag] = Class

        self._set_internal_attr("_cached_imported_classes", dict_classes)
        return dict_classes

    def complete_with_classes(self):
        """Populate info solver by executing ``_complete_info_solver`` class
        methods
        """
        dict_classes = self.import_classes()
        for Class in list(dict_classes.values()):
            if hasattr(Class, "_complete_info_solver"):
                Class._complete_info_solver(self)
