"""Base solver (:mod:`fluidsim.base.solvers.base`)
==================================================

.. currentmodule:: fluidsim.base.solvers.base

Provides:

.. autoclass:: InfoSolverBase
   :members:
   :private-members:

.. autoclass:: SimulBase
   :members:
   :private-members:

"""

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
                   'FORCING': False,
                   # Physical parameters:
                   'nu_2': 0.}
        params._set_attribs(attribs)

    @classmethod
    def create_default_params(cls):
        cls.info_solver = cls.InfoSolver()
        cls.info_solver.complete_with_classes()
        return create_params(cls.info_solver)

    def __init__(self, params):
        # np.seterr(invalid='raise')
        # np.seterr(over='raise')
        np.seterr(all='warn')
        np.seterr(under='ignore')

        if not hasattr(self, 'info_solver'):
            self.info_solver = self.InfoSolver()
            self.info_solver.complete_with_classes()

        dico_classes = self.info_solver.import_classes()

        if not isinstance(params, Parameters):
            raise TypeError('params should be a Parameters instance.')

        # params.check_and_modify()
        self.params = params
        self.info = create_info_simul(self.info_solver, params)

        # initialization operators and grid
        Operators = dico_classes['Operators']
        self.oper = Operators(params=params)

        # initialization output
        Output = dico_classes['Output']
        self.output = Output(self)

        self.output.print_stdout(
            '*************************************\n' +
            'Program FluidDyn')

        # output.print_memory_usage(
        #     'Memory usage after creating operator (equiv. seq.)')

        # initialisation object variables
        State = dico_classes['State']
        self.state = State(self)

        # initialisation time stepping
        TimeStepping = dico_classes['TimeStepping']
        self.time_stepping = TimeStepping(self)

        # initialisation fields (and time if needed)
        InitFields = dico_classes['InitFields']
        self.init_fields = InitFields(self)
        self.init_fields()

        # just for the first output
        if params.time_stepping.USE_CFL:
            self.time_stepping._compute_time_increment_CLF()

        # initialisation forcing
        if params.FORCING:
            Forcing = dico_classes['Forcing']
            self.forcing = Forcing(self)
            self.forcing.compute()

        # complete the initialisation of the object output
        self.output.init_with_oper_and_state()
        
        # if enabled, preprocesses flow parameters such as viscosity and forcing
        # based on initialized fields
        Preprocess = dico_classes['Preprocess']
        self.preprocess = Preprocess(self)
        self.preprocess()

    def tendencies_nonlin(self, variables=None):
        """Return a null SetOfVariables object."""
        tendencies = SetOfVariables(
            like=self.state.state_fft,
            info='tendencies_nonlin')
        tendencies.initialize(value=0.)
        return tendencies


Simul = SimulBase


if __name__ == "__main__":

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'
