"""Base solver (:mod:`fluidsim.base.solvers.info_base`)
=======================================================

.. autoclass:: InfoSolverBase
   :members:
   :private-members:

"""
from copy import deepcopy

from fluidsim_core.info import InfoSolverCore, create_info_simul  # noqa


def _merged_element(el1, el2):
    result = deepcopy(el1)
    result.extend(deepcopy(el2))
    return result


class InfoSolverBase(InfoSolverCore):
    """Contain the information on a solver."""

    def _init_root(self):
        super()._init_root()

        self._set_attribs(
            {
                "module_name": "fluidsim.base.solvers.base",
                "class_name": "SimulBase",
                "short_name": "Base",
            }
        )

        classes = self.classes

        classes._set_child(
            "Operators",
            attribs={
                "module_name": "fluidsim.operators.operators0d",
                "class_name": "Operators0D",
            },
        )

        classes._set_child(
            "State",
            attribs={
                "module_name": "fluidsim.base.state",
                "class_name": "StateBase",
            },
        )

        classes._set_child(
            "TimeStepping",
            attribs={
                "module_name": "fluidsim.base.time_stepping.simple",
                "class_name": "TimeSteppingSimple",
            },
        )

        classes._set_child(
            "InitFields",
            attribs={
                "module_name": "fluidsim.base.init_fields",
                "class_name": "InitFieldsBase",
            },
        )

        classes._set_child(
            "Forcing",
            attribs={
                "module_name": "fluidsim.base.forcing",
                "class_name": "ForcingBase",
            },
        )

        classes._set_child(
            "Output",
            attribs={
                "module_name": "fluidsim.base.output.base",
                "class_name": "OutputBase",
            },
        )
