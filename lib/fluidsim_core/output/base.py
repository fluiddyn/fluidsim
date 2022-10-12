"""Output API

.. autoclass:: SimReprMakerCore
   :members:
   :private-members:

.. autoclass:: OutputCore
   :members:
   :private-members:
   :special-members: __init__

"""
from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path
from time import sleep

import fluiddyn
from fluiddyn.io import FLUIDDYN_PATH_SCRATCH, FLUIDSIM_PATH
from fluiddyn.util import time_as_str, mpi

from fluidsim_core import __version__


class SimReprMakerCore:
    """Code to create strings to represent a simulation"""

    def __init__(self, sim):
        self.sim = sim
        self.params = sim.params
        self.ordered_keys = []
        self.parameters = {}
        self.formats = {}

    def add_word(self, word):
        if word:
            self.ordered_keys.append(("__word", word))

    def add_parameters(self, parameters, formats=None, indices=None):
        for key in parameters:
            if indices and key in indices:
                index = indices[key]
                self.ordered_keys.insert(index, ("__parameter", key))
            else:
                self.ordered_keys.append(("__parameter", key))
        self.parameters.update(parameters)
        if formats:
            self.formats.update(formats)

    def _make_list_repr(self):
        list_repr = []
        for kind, value in self.ordered_keys:
            if kind == "__word":
                list_repr.append(value)
            elif kind == "__parameter":
                name_parameter = value
                parameter = self.parameters[name_parameter]
                if isinstance(parameter, float):
                    default_fmt = ".3f"
                else:
                    default_fmt = ""
                fmt = self.formats.get(name_parameter, default_fmt)
                str_parameter = ("{:" + fmt + "}").format(parameter)
                if fmt and fmt[-1] in ["e", "g"] and "e+" in str_parameter:
                    str_parameter = str_parameter.replace("e+", "e")
                if "e" not in str_parameter and "." in str_parameter:
                    str_parameter = str_parameter.rstrip("0")
                if str_parameter.endswith("."):
                    str_parameter = str_parameter[:-1]
                list_repr.append((name_parameter, str_parameter))
            else:
                raise ValueError
        return list_repr

    def get_list_repr(self):
        if not hasattr(self, "_list_repr"):
            self._list_repr = self._make_list_repr()
        return self._list_repr

    def time_from_path_run(self, path_run):
        if isinstance(path_run, Path):
            path_run = path_run.name
        return "_".join(path_run.split("_")[-2:])

    def get_time_as_str(self):
        params = self.sim.params
        if not params.NEW_DIR_RESULTS:
            return self.time_from_path_run(params.path_run)
        else:
            return time_as_str()

    def make_representations(self):
        """Generate a unique name and summary for the simulation run"""
        time = self.get_time_as_str()
        list_repr = self.get_list_repr().copy()
        list_repr.append(time)

        for_name = []
        for_summary = []
        for obj in list_repr:
            if isinstance(obj, tuple):
                name_parameter, str_parameter = obj
                obj = f"{name_parameter}{str_parameter}"
                str_summary = f"{name_parameter}={str_parameter}"
            else:
                str_summary = obj.replace("_", ", ")

            for_name.append(obj)
            for_summary.append(str_summary)

        name_run = "_".join(for_name)
        summary_simul = ", ".join(for_summary)
        return name_run, summary_simul


class OutputCore(ABC):
    """Base Output class"""

    SimReprMaker = SimReprMakerCore

    @abstractstaticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver. More *specialized output*
        classes may be assigned as follows:

        .. code-block:: python

            classes = info_solver.classes.Output.classes
            classes._set_child(
                "PrintStdOut",
                attribs={
                    "module_name": "package.module",
                    "class_name": "MyPrintStdOut",
                },
            )

        """
        info_solver.classes.Output._set_child("classes")

    @abstractstaticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container."""
        attribs = {
            "HAS_TO_SAVE": True,
            "sub_directory": "",
            # Possibly more ...
        }
        params._set_child("output", attribs=attribs)
        # params.output._set_doc("...")

    @abstractmethod
    def __init__(self, sim):
        """Initializes the instance attributes and child *output* classes.

        .. note::
            This class relies on a boolean parameter ``params.NEW_DIR_RESULTS``.

        """
        #: The simulation class
        self.sim = sim

        if hasattr(sim, "oper"):
            #: An alias towards the Operators object
            self.oper = sim.oper

        params = sim.params
        #: The tree of parameters for output-related classes
        self.params = params.output

        #: Determines whether to save output files on-the-fly
        self._has_to_save = self.params.HAS_TO_SAVE
        #: Alias for solver short-name
        self.name_solver = sim.info.solver.short_name

        # initialisation name_run and path_run
        if mpi.rank == 0:
            self._init_name_run()
        else:
            self.name_run = None

        if not params.NEW_DIR_RESULTS:
            try:
                self.path_run = str(params.path_run)
            except AttributeError:
                params.NEW_DIR_RESULTS = True
                print(
                    "Strange: params.NEW_DIR_RESULTS == False "
                    "but no params.path_run"
                )

            # if _has_to_save, we verify the correspondence between the
            # resolution of the simulation and the resolution of the
            # previous simulation saved in this directory
            if self._has_to_save:
                if mpi.rank == 0:
                    try:
                        params_dir = sim.Parameters._load_params_simul(
                            self.path_run
                        )
                    except IOError:
                        raise ValueError(
                            "Strange, no info_simul.h5 in self.path_run"
                        )

                    cond = False
                    for n in ("nx", "ny", "nz"):
                        try:
                            # Mismatch in resolution
                            if getattr(params.oper, n) != getattr(
                                params_dir.oper, n
                            ):
                                cond = True
                                break
                        except AttributeError:
                            pass

                    if cond:
                        params.NEW_DIR_RESULTS = True
                        print(
                            """
Warning: params.NEW_DIR_RESULTS is False but the resolutions of the simulation
         and of the simulation in the directory self.path_run are different
         we put params.NEW_DIR_RESULTS = True"""
                        )
                if mpi.nb_proc > 1:
                    params.NEW_DIR_RESULTS = mpi.comm.bcast(
                        params.NEW_DIR_RESULTS
                    )

        if params.NEW_DIR_RESULTS:
            #: Full path of the output directory for this specific simulation
            self.path_run = self._init_path_run()

        if mpi.nb_proc > 1:
            # ensure same name_run across all processes
            self.name_run = mpi.comm.bcast(self.name_run, root=0)

        self.sim.name_run = self.name_run

        #: Optionally an instance of class PrintStdout which writes output
        #  to stdout and also a file buffer
        self.print_stdout = print

    def _init_path_run(self):
        """Initialize a unique path for the simulation.

        Returns
        -------
        str

        """
        params = self.sim.params

        if FLUIDDYN_PATH_SCRATCH is not None:
            path_base = FLUIDDYN_PATH_SCRATCH
        else:
            path_base = FLUIDSIM_PATH

        path_base = Path(path_base)

        if len(params.output.sub_directory) > 0:
            path_base = path_base / params.output.sub_directory

        if mpi.rank == 0:
            while True:
                path_run = path_base / self.name_run
                if not params.output.HAS_TO_SAVE:
                    break
                if not path_run.exists():
                    try:
                        path_run.mkdir(parents=True)
                    except OSError:
                        # in case of simultaneously launched simulations
                        print(
                            'Warning: NEW_DIR_RESULTS=True, but path"',
                            path_run,
                            "already exists. Trying a new path...",
                        )
                        sleep(1)
                        self._init_name_run()
                    else:
                        break
                else:
                    sleep(1)
                    self._init_name_run()

            path_run = str(path_run)
        else:
            path_run = None

        if mpi.rank == 0:
            params._set_attrib("path_run", path_run)
        if mpi.nb_proc > 1:
            path_run = mpi.comm.bcast(path_run, root=0)

        return path_run

    def _init_sim_repr_maker(self):
        """Create a list of strings to make the run name.

        Returns
        -------
        :any:`fluidsim_core.output.base.SimReprMakerCore`

        """
        sim = self.sim

        sim_repr_maker = self.SimReprMaker(sim)
        sim_repr_maker.add_word(self.name_solver)
        sim_repr_maker.add_word(sim.params.short_name_type_run)

        # Modify sim_repr_maker as necessary
        return sim_repr_maker

    def _init_name_run(self):
        """Initialize the ``name_run`` and ``summary_simul`` attributes by
        calling :any:`fluidsim_core.output.base.SimReprMakerCore.make_representations`

        """
        if not hasattr(self, "_sim_repr_maker"):
            self._sim_repr_maker = self._init_sim_repr_maker()
        (
            #: Name of the output directory for this specific simulation
            self.name_run,
            self.summary_simul,
        ) = self._sim_repr_maker.make_representations()

    @abstractmethod
    def post_init(self):
        """Execute once the sim object is injected with all child classes.
        Typically used to print descriptive initialization messages.

        """
        sim = self.sim

        if mpi.rank == 0:
            objects_to_print = {
                "sim": sim,
                "sim.output": sim.output,
            }

            for key, obj in objects_to_print.items():
                self.print_stdout(
                    "{:20s}".format(key + ": ") + str(obj.__class__)
                )
        self._save_info_solver_params_xml()

    def _save_info_solver_params_xml(self, replace=False, comment=""):
        """Save files with information on the solver and on the run."""
        if (
            mpi.rank == 0
            and self._has_to_save
            and self.sim.params.NEW_DIR_RESULTS
        ):
            comment = f"""\
This file should not be modified (except for adding xml comments).
Created by the Python programs:
FluidDyn {fluiddyn.__version__}
FluidSim Core {__version__}
{comment}
"""

            path_run = Path(self.path_run)
            info_solver_xml_path = path_run / "info_solver.xml"
            params_xml_path = path_run / "params_simul.xml"

            # save info on the run
            if replace:
                info_solver_xml_path.unlink()
                params_xml_path.unlink()

            self.sim.info.solver._save_as_xml(
                path_file=info_solver_xml_path, comment=comment
            )

            self.sim.params._save_as_xml(
                path_file=params_xml_path, comment=comment
            )
