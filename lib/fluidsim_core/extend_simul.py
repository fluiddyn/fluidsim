"""Mechanism to extend a Simul class with just a simple class
=============================================================

.. autofunction:: extend_simul_class

.. autoclass:: SimulExtender
   :members:
   :private-members:

"""

from logging import warn
from pathlib import Path

import h5py

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util import import_class
from fluiddyn.util import mpi


def extend_simul_class(Simul, extenders):
    """Extend a Simul class with "Simul extenders".

    "Simul extenders" are classes deriving from
    :class:`fluidsim.extend_simul.SimulExtender`.

    """

    if isinstance(extenders, type):
        extenders = [extenders]

    class NewInfoSolver(Simul.InfoSolver):
        pass

    if not hasattr(NewInfoSolver, "_modificators"):
        NewInfoSolver._modificators = []
        NewInfoSolver._extenders = []

    for extender in extenders:
        modif_info_solver = extender.get_modif_info_solver()
        NewInfoSolver._modificators.append(modif_info_solver)
        NewInfoSolver._extenders.append(extender)

    class NewSimul(Simul):
        InfoSolver = NewInfoSolver

    return NewSimul


class SimulExtender:
    """Abstract class to define a "Simul extender"

    Simul extenders are classes that can extend a ``Simul`` class to change its
    behavior for some simulations.

    This class is meant to be subclassed. The child class has to contain one
    class attribute ``_module_name`` and two class methods
    ``get_modif_info_solver`` and ``complete_params_with_default``. An example
    can be found in the module
    :mod:`fluidsim.extend_simul.spatial_means_regions_milestone`.

    """

    @classmethod
    def get_modif_info_solver(cls):
        """Create a function to modify ``info_solver``.

        Note that this function is called when the object ``info_solver`` has
        not yet been created (and cannot yet be modified)! This is why one
        needs to create a function that will be called later to modify
        ``info_solver``.

        """
        raise NotImplementedError

    @classmethod
    def complete_params_with_default(cls, params):
        """Should complete the simul parameters"""
        raise NotImplementedError

    @classmethod
    def _complete_params_with_default(cls, params):
        cls.complete_params_with_default(params)


def _extend_simul_class_from_path(Simul, path_file):
    """Extend a Simul if needed from a path file (internal API)."""

    path_file = Path(path_file)

    if mpi.rank == 0:
        if path_file.suffix == ".xml":
            # we assume that it is a info_solver.xml file
            info_solver = ParamContainer(path_file=path_file)
            if hasattr(info_solver, "extenders"):
                extenders = info_solver.extenders
            else:
                extenders = []
        else:
            with h5py.File(path_file, "r") as file:
                extenders = list(
                    file["/info_simul/solver"].attrs.get("extenders", [])
                )
    else:
        extenders = None

    if mpi.nb_proc > 1:
        extenders = mpi.comm.bcast(extenders)

    extender_classes = []

    for extender_full_name in extenders:
        module_name, class_name = extender_full_name.rsplit(".", 1)

        try:
            extender_class = import_class(module_name, class_name)
        except ImportError:
            warn(f"ImportError extender class {extender_full_name}.")
        else:
            extender_classes.append(extender_class)

    if extender_classes:
        return extend_simul_class(Simul, extender_classes)
    else:
        return Simul
