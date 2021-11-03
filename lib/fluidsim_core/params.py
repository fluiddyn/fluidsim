"""Solver parameters

.. autofunction:: iter_complete_params

.. autoclass:: Parameters
   :members:
   :private-members:

"""
import os
from glob import glob
from pathlib import Path

import h5py
from fluiddyn.util import import_class, mpi
from fluiddyn.util.paramcontainer import ParamContainer

from .info import InfoSolverCore


def iter_complete_params(params, info_solver, classes):
    """Iterate over a collection of classes and invoke the static method
    ``_complete_params_with_default`` to populate an instance of
    :class:`Parameters` in-place.

    Parameters
    ----------
    params : :class:`fluidsim_core.params.Parameters` or its subclass

    info_solver : :class:`fluidsim_core.info.InfoSolverCore` or its subclass

    classes : iterable

    """
    for Class in classes:
        if hasattr(Class, "_complete_params_with_default"):
            try:
                Class._complete_params_with_default(params)
            except TypeError:
                try:
                    Class._complete_params_with_default(params, info_solver)
                except TypeError as e:
                    e.args += ("for class: " + repr(Class),)
                    raise


class Parameters(ParamContainer):
    """Contain the parameters."""

    @classmethod
    def _create_params(cls, input_info_solver):
        """Create a Parameters instance from an InfoSolverCore instance."""
        if isinstance(input_info_solver, InfoSolverCore):
            info_solver = input_info_solver
        elif hasattr(input_info_solver, "Simul"):
            info_solver = input_info_solver.Simul.create_default_params()
        else:
            raise ValueError(
                "Can not create params from input input_info_solver."
            )

        params = cls(tag="params")

        dict_classes = {
            "Solver": import_class(
                info_solver.module_name, info_solver.class_name
            )
        }
        dict_classes.update(info_solver.import_classes())

        iter_complete_params(params, info_solver, dict_classes.values())
        return params

    @classmethod
    def _load_params_simul(cls, path=None, only_mpi_rank0=True):
        """Load the parameters and return a Parameters instance."""
        if mpi.rank > 0 and not only_mpi_rank0:
            params = None
        else:
            if path is None:
                path = os.getcwd()

            path_xml = None
            if os.path.isdir(path):
                path_xml = os.path.join(path, "params_simul.xml")
            elif path.endswith(".xml"):
                if not os.path.exists(path):
                    raise ValueError("The file " + path + "does not exists.")

                path_xml = path

            if path_xml is not None and os.path.exists(path_xml):
                params = cls(path_file=path_xml)
            else:
                if os.path.isfile(path):
                    paths = [path]
                else:
                    paths = glob(os.path.join(path, "state_*"))
                if paths:
                    path = sorted(paths)[0]

                    if len(path) > 100:
                        str_path = "[...]" + path[-100:]
                    else:
                        str_path = path

                    print("Loading params from file\n" + str_path)
                    with h5py.File(path, "r") as h5file:
                        params = cls(hdf5_object=h5file["/info_simul/params"])
                else:
                    raise ValueError

        if mpi.nb_proc > 1 and not only_mpi_rank0:
            params = mpi.comm.bcast(params, root=0)
        return params

    @classmethod
    def _load_info_solver(cls, path_dir=None):
        """Load the solver information, return an InfoSolverCore instance."""
        if path_dir is None:
            path_dir = os.getcwd()

        if not isinstance(path_dir, Path):
            path_dir = Path(path_dir)

        if not path_dir.is_dir():
            raise ValueError(str(path_dir) + " is not a directory")

        path_info_solver = path_dir / "info_solver.xml"
        if path_info_solver.exists():
            return cls(path_file=str(path_info_solver))

        paths = sorted(path_dir.glob("state_*"))

        if not paths:
            raise ValueError("No result files in dir " + str(path_dir))

        path = str(paths[0])

        if len(path) > 100:
            str_path = "[...]" + path[-100:]
        else:
            str_path = path

        mpi.printby0("load params from file\n" + str_path)
        with h5py.File(path, "r") as h5file:
            return cls(hdf5_object=h5file["/info_simul/solver"])

    def __ior__(self, other):
        """Defines operator `|=`

        This operator performs a union with other Parameters instances.
        In simpler words, merge missing attributes and children (members).
        Do note, this does not check for mismatch in the parameter values.

        Parameters
        ----------
        other: Parameters
            Another Parameters instance to look for missing members.

        Returns
        -------
        Parameters

        """
        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(other)}. Can only merge instances of Parameters."
            )

        params1 = self
        params2 = other

        # Merge attributes
        diff_attribs = set(params2._key_attribs) - set(params1._key_attribs)

        if len(diff_attribs) > 0:
            print("Add parameter attributes: ", diff_attribs)

        for attrib in diff_attribs:
            params1._set_attrib(attrib, params2[attrib])

        # Merge children
        diff_children = set(params2._tag_children) - set(params1._tag_children)
        internal_attribs = [
            "attribs",
            "children",
            "key_attribs",
            "tag",
            "tag_children",
        ]

        if len(diff_children) > 0:
            print("Add parameter children: ", diff_children)

        for child in diff_children:
            child_attribs = params2[child]._make_dict()
            # Clean up internal attributes from dictionary
            list(map(child_attribs.__delitem__, internal_attribs))

            params1._set_child(child, child_attribs)

        # Recursive
        for child in params2._tag_children:
            params1[child] |= params2[child]

        return self
