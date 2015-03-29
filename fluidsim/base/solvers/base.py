"""Base solver (:mod:`fluidsim.base.solvers.base`)
========================================================

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


from fluidsim.operators.setofvariables import SetOfVariables

from fluidsim.base.params import Parameters

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

    @staticmethod
    def _complete_params_with_default(params):
        """A static method used to complete the *params* container."""
        attribs = {'short_name_type_run': '',
                   'NEW_DIR_RESULTS': True,
                   'ONLY_COARSE_OPER': False,
                   'FORCING': False,
                   # Physical parameters:
                   'nu_2': 0.}
        params.set_attribs(attribs)

    def __init__(self, params, info_solver=None):
        # np.seterr(invalid='raise')
        # np.seterr(over='raise')
        np.seterr(all='warn')
        np.seterr(under='ignore')

        if info_solver is None:
            info_solver = InfoSolverBase()
            info_solver.complete_with_classes()
        elif not isinstance(info_solver, InfoSolverBase):
            raise ValueError('info_solver must be an InfoSolverBase object.')
        dico_classes = info_solver.import_classes()

        if not isinstance(params, Parameters):
            raise TypeError('params should be a Parameters instance.')

        # params.check_and_modify()
        self.params = params
        self.info = create_info_simul(info_solver, params)

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
        self.state = State(self, info_solver)

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
            self.forcing = Forcing(params, self)
            self.forcing.compute()

        # complete the initialisation of the object output
        self.output.init_with_oper_and_state()

    def tendencies_nonlin(self, variables=None):
        """Return a null SetOfVariables object."""
        tendencies = SetOfVariables(
            like_this_sov=self.state.state_fft,
            name_type_variables='tendencies_nonlin')
        tendencies.initialize(value=0.)
        return tendencies


Simul = SimulBase


if __name__ == "__main__":

    import fluiddyn as fld

    info_solver = InfoSolverBase()
    info_solver.complete_with_classes()

    params = fld.simul.create_params(info_solver)

    params.short_name_type_run = 'test'

    nh = 16
    Lh = 2*np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    delta_x = params.oper.Lx/params.oper.nx
    params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8

    params.time_stepping.t_end = 5.

    params.init_fields.type_flow_init = 'NOISE'

    params.output.periods_plot.phys_fields = 0.

    params.output.periods_print.print_stdout = 0.25
    params.output.periods_save.phys_fields = 2.

    sim = Simul(params)

    sim.output.phys_fields.plot()
    sim.time_stepping.start()
    sim.output.phys_fields.plot()

    fld.show()
