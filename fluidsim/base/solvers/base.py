"""Base solver (:mod:`fluidsim.base.solvers.base`)
==================================================

Provides:

.. autoclass:: InfoSolverBase
   :members:
   :private-members:

.. autoclass:: SimulBase
   :members:
   :private-members:

"""
from time import time
import atexit
from pathlib import Path
from warnings import warn
import signal
import os

import numpy as np

from fluiddyn import time_as_str
from fluiddyn.util import mpi
from fluidsim_core.solver import SimulCore

from ..setofvariables import SetOfVariables
from .info_base import InfoSolverBase


class SimulBase(SimulCore):
    """Represent a solver.

    This is the main base class which is inherited by the other
    simulation classes.

    A :class:`SimulBase` object contains at least one object of the
    classes:

    - :class:`fluidsim.base.params.Parameters`
    - :class:`fluidsim.base.time_stepping.TimeSteppingBase`
    - :class:`fluidsim.operators.operators.Operators`
    - :class:`fluidsim.base.state.StateBase`

    Parameters
    ----------

    params : :class:`fluidsim.base.params.Parameters`
        Parameters for the simulation.

    info_solver : :class:`fluidsim.base.solvers.info_base.InfoSolverBase`
        Information about the particular solver.

    """

    InfoSolver = InfoSolverBase

    @staticmethod
    def _complete_params_with_default(params):
        """A static method used to complete the *params* container."""
        attribs = {
            "short_name_type_run": "",
            "NEW_DIR_RESULTS": True,
            "ONLY_COARSE_OPER": False,
            # Physical parameters:
            "nu_2": 0.0,
        }
        params._set_attribs(attribs)
        params._set_doc(
            """
short_name_type_run: str

    A short name of the simulation used to create the directory name.

NEW_DIR_RESULTS: bool

    To be used only when loading a simulation. If True (default), a new directory
    is created to contain the results of the simulation. If False, the results of
    the simulation are appended in the old directory.

ONLY_COARSE_OPER: bool

    To be used only when loading a simulation. If True (not default), the operator
    is created with a very small resolution. It is very fast but then it can not
    be used to process data.

nu_2: float (default = 0.)

    Viscosity coefficient. Used in particular in the method
    :func:`fluidsim.base.solvers.pseudo_spect.SimulBasePseudoSpectral.compute_freq_diss`).

"""
        )

    @classmethod
    def create_default_params(cls):
        return super().create_default_params()

    def __enter__(self):
        if not hasattr(self, "_end_of_simul") or self._end_of_simul:
            self.time_stepping._time_beginning_simul = time()
            self._end_of_simul = False
        return self

    def __exit__(self, *args):
        if not self._end_of_simul:
            total_time_simul = time() - self.time_stepping._time_beginning_simul
            self.time_stepping.time_simul_in_sec = total_time_simul
            self.output.end_of_simul(total_time_simul)
            self._end_of_simul = True

        if self._lockfile is not None and self._lockfile.exists():
            self._lockfile.unlink()

    def __init__(self, params):
        super().__init__(params)

        # np.seterr(invalid='raise')
        # np.seterr(over='raise')
        np.seterr(all="warn")
        np.seterr(under="ignore")

        dict_classes = self.info_solver.import_classes()

        # initialization operators and grid
        Operators = dict_classes["Operators"]
        self.oper = Operators(params=params)

        # initialization output
        Output = dict_classes["Output"]
        self.output = Output(self)

        if mpi.rank == 0 and self.params.output.HAS_TO_SAVE:
            self._lockfile = Path(self.output.path_run) / "is_being_advanced.lock"

            if self._lockfile.exists():
                warn(
                    f"File {self._lockfile} already exists. It could "
                    "mean that this simulation is already being advanced by "
                    "another process or that an old lockfile has not been "
                    "deleted (which is a bug). If no process is advancing "
                    "the simulation, the lockfile can safely be removed."
                )
            else:
                with open(self._lockfile, "w") as file:
                    file.write(time_as_str() + f"\n{os.getpid()}\n")

            def release_lock():
                if self._lockfile.exists():
                    self._lockfile.unlink()

            atexit.register(release_lock)

            def sig_handler(signo, frame):
                release_lock()
                if signo == signal.SIGINT:
                    signal.default_int_handler(signo, frame)
                elif signo == signal.SIGTERM:
                    self.time_stepping._stop_signal_received = signo

            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, sig_handler)

        else:
            self._lockfile = None

        self.output.print_stdout(
            "*************************************\nProgram fluidsim"
        )

        # output.print_memory_usage(
        #     'Memory usage after creating operator (equiv. seq.)')

        # initialisation object variables
        State = dict_classes["State"]
        self.state = State(self)

        # initialisation time stepping
        TimeStepping = dict_classes["TimeStepping"]
        self.time_stepping = TimeStepping(self)

        # initialisation fields (and time if needed)
        InitFields = dict_classes["InitFields"]
        self.init_fields = InitFields(self)
        self.init_fields()

        # initialisation forcing and potential other classes (like turb_model)
        keys_simple_classes = (
            "Operators",
            "Output",
            "State",
            "TimeStepping",
            "InitFields",
            "Preprocess",
        )
        other_classes = {
            key: cls
            for key, cls in dict_classes.items()
            if key not in keys_simple_classes
        }
        for cls in other_classes.values():
            name_task = cls._name_task
            setattr(self, f"is_{name_task}_enabled", False)
            try:
                params_cls = getattr(params, name_task)
            except AttributeError:
                pass
            else:
                if params_cls.enable:
                    setattr(self, f"is_{name_task}_enabled", True)
                    setattr(self, name_task, cls(self))

        # complete the initialisation of the object output
        self.output.post_init()

        # if enabled, preprocesses flow parameters such as viscosity and
        # forcing based on initialized fields
        if "Preprocess" in dict_classes:
            Preprocess = dict_classes["Preprocess"]
            self.preprocess = Preprocess(self)
            self.preprocess()

    def tendencies_nonlin(self, variables=None, old=None):
        """Return a null SetOfVariables object."""
        if old is None:
            tendencies = SetOfVariables(
                like=self.state.state_phys, info="tendencies_nonlin"
            )
        else:
            tendencies = old
        tendencies.initialize(value=0.0)
        return tendencies


Simul = SimulBase


if __name__ == "__main__":

    params = Simul.create_default_params()

    params.short_name_type_run = "test"
    params.time_stepping.USE_CFL = False
    params.time_stepping.t_end = 2.0
    params.time_stepping.deltat0 = 0.1

    sim = Simul(params)
    sim.time_stepping.start()
