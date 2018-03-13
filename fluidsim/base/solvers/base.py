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

from builtins import object
from warnings import warn
import numpy as np


from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.base.params import Parameters, create_params

from fluidsim.base.solvers.info_base import (
    InfoSolverBase, create_info_simul)


class SimulBase(object):
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
        attribs = {'short_name_type_run': '',
                   'NEW_DIR_RESULTS': True,
                   'ONLY_COARSE_OPER': False,
                   # Physical parameters:
                   'nu_2': 0.}
        params._set_attribs(attribs)

    @classmethod
    def create_default_params(cls):
        cls.info_solver = cls.InfoSolver()
        cls.info_solver.complete_with_classes()
        return create_params(cls.info_solver)

    def __enter__(self):
        if not hasattr(self, '_end_of_simul') or self._end_of_simul:
            self.time_stepping._time_beginning_simul = time()
            self._end_of_simul = False

        return self

    def __exit__(self, *args):
        if not self._end_of_simul:
            total_time_simul = (
                time() - self.time_stepping._time_beginning_simul)
            self.time_stepping.time_simul_in_sec = total_time_simul
            self.output.end_of_simul(total_time_simul)
            self._end_of_simul = True

    def __init__(self, params):
        # np.seterr(invalid='raise')
        # np.seterr(over='raise')
        np.seterr(all='warn')
        np.seterr(under='ignore')

        if not hasattr(self, 'info_solver') or \
           self.info_solver.__class__ is not self.InfoSolver:
            warn("Creating a new info_solver instance because it's missing or "
                 "due to type mismatch  {}".format(self.InfoSolver))
            self.info_solver = self.InfoSolver()
            self.info_solver.complete_with_classes()

        dict_classes = self.info_solver.import_classes()

        if not isinstance(params, Parameters):
            raise TypeError('params should be a Parameters instance.')

        self.params = params
        self.info = create_info_simul(self.info_solver, params)

        # initialization operators and grid
        Operators = dict_classes['Operators']
        self.oper = Operators(params=params)

        # initialization output
        Output = dict_classes['Output']
        self.output = Output(self)

        self.output.print_stdout(
            '*************************************\n' +
            'Program fluidsim')

        # output.print_memory_usage(
        #     'Memory usage after creating operator (equiv. seq.)')

        # initialisation object variables
        State = dict_classes['State']
        self.state = State(self)

        # initialisation time stepping
        TimeStepping = dict_classes['TimeStepping']
        self.time_stepping = TimeStepping(self)

        # initialisation fields (and time if needed)
        InitFields = dict_classes['InitFields']
        self.init_fields = InitFields(self)
        self.init_fields()

        # just for the first output
        if hasattr(params.time_stepping, 'USE_CFL') and \
           params.time_stepping.USE_CFL:
            self.time_stepping._compute_time_increment_CLF()

        # initialisation forcing
        self.is_forcing_enabled = False
        try:
            params.forcing
        except AttributeError:
            pass
        else:
            if params.forcing.enable:
                self.is_forcing_enabled = True
                Forcing = dict_classes['Forcing']
                self.forcing = Forcing(self)
                self.forcing.compute()

        # complete the initialisation of the object output
        self.output.init_with_oper_and_state()

        # if enabled, preprocesses flow parameters such as viscosity and
        # forcing based on initialized fields
        if 'Preprocess' in dict_classes:
            Preprocess = dict_classes['Preprocess']
            self.preprocess = Preprocess(self)
            self.preprocess()

    def tendencies_nonlin(self, variables=None, old=None):
        """Return a null SetOfVariables object."""
        if old is None:
            tendencies = SetOfVariables(
                like=self.state.state_phys,
                info='tendencies_nonlin')
        else:
            tendencies = old
        tendencies.initialize(value=0.)
        return tendencies


Simul = SimulBase


if __name__ == "__main__":

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'
    params.time_stepping.USE_CFL = False
    params.time_stepping.t_end = 2.
    params.time_stepping.deltat0 = 0.1

    sim = Simul(params)
    sim.time_stepping.start()
