"""Forcing schemes (:mod:`fluidsim.base.forcing.base`)
============================================================


Provides:

.. autoclass:: ForcingBase
   :members:
   :private-members:

.. autoclass:: ForcingBasePseudoSpectral
   :members:
   :private-members:

"""

from builtins import object

import numpy as np

from fluiddyn.util import mpi

from .specific import (
    InScriptForcingPseudoSpectral,
    InScriptForcingPseudoSpectralCoarse,
    Proportional,
    TimeCorrelatedRandomPseudoSpectral,
)


class ForcingBase:
    """Organize the forcing schemes (base class)"""

    @staticmethod
    def _complete_info_solver(info_solver, classes=None):
        """Complete the ParamContainer info_solver."""
        info_solver.classes.Forcing._set_child("classes")

        if classes is not None:
            classesXML = info_solver.classes.Forcing.classes

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
        """This static method is used to complete the *params* container.
        """
        params._set_child(
            "forcing",
            attribs={
                "enable": False,
                "type": "",
                "available_types": [],
                "forcing_rate": 1.0,
                "key_forced": None,
            },
        )

        params.forcing._set_doc(
            """

See :mod:`fluidsim.base.forcing`.

type : str

  Type of forcing.

available_types : list

  Available types that can be used.

forcing_rate : float

  Forcing injection rate.

key_forced: {None} or str

  The key of the variable to be forced. If it is None, a default key depending
  of the type of forcing is used.

"""
        )

        dict_classes = info_solver.classes.Forcing.import_classes()
        # iter on the dict in a determined order
        for key in sorted(dict_classes.keys()):
            cls = dict_classes[key]
            if hasattr(cls, "_complete_params_with_default"):
                cls._complete_params_with_default(params)

    def __init__(self, sim):
        self.type_forcing = sim.params.forcing.type

        dict_classes = sim.info.solver.classes.Forcing.import_classes()

        if self.type_forcing not in dict_classes:

            # temporary trick to open old simulations
            if self.type_forcing == "random" and "tcrandom" in dict_classes:
                self.type_forcing = "tcrandom"
                sim.params.forcing.__dict__["tcrandom"] = sim.params.forcing[
                    "random"
                ]
            else:
                if mpi.rank == 0:
                    print("dict_classes:", dict_classes)
                raise ValueError(
                    "Wrong value for params.forcing.type: " + self.type_forcing
                )

        ClassForcing = dict_classes[self.type_forcing]

        self.sim = sim
        if not sim.params.ONLY_COARSE_OPER:
            self.forcing_maker = ClassForcing(sim)
        else:
            self.forcing_maker = None

        self._t_last_computed = -np.inf

    def __call__(self, key):
        """Return the variable corresponding to the given key."""
        forcing = self.get_forcing()
        return forcing.get_var(key)

    def compute(self):
        time = self.sim.time_stepping.t
        if time > self._t_last_computed:
            self.forcing_maker.compute()
            self._t_last_computed = time

    def get_forcing(self):
        return self.forcing_maker.forcing_phys

    def is_initialized(self):
        if hasattr(self.forcing_maker, "is_initialized"):
            return self.forcing_maker.is_initialized

        else:
            return True


class ForcingBasePseudoSpectral(ForcingBase):
    """Organize the forcing schemes (pseudo-spectra)

    .. inheritance-diagram:: ForcingBasePseudoSpectral

    """

    @staticmethod
    def _complete_info_solver(info_solver, classes=None):
        """Complete the ParamContainer info_solver."""

        classes_default = (
            InScriptForcingPseudoSpectral,
            InScriptForcingPseudoSpectralCoarse,
            Proportional,
            TimeCorrelatedRandomPseudoSpectral,
        )

        if classes is None:
            classes = classes_default[:]
        else:
            for cls_default in classes_default:
                if not any(cls_default.tag == cls.tag for cls in classes):
                    classes.append(cls_default)

        ForcingBase._complete_info_solver(info_solver, classes=classes)

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        ForcingBase._complete_params_with_default(params, info_solver)

        params.forcing._set_attribs({"nkmax_forcing": 5, "nkmin_forcing": 4})

    def get_forcing(self):
        return self.forcing_maker.forcing_fft
