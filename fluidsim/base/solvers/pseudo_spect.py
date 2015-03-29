"""Base solver (:mod:`fluidsim.base.solvers.pseudo_spect`)
================================================================

.. currentmodule:: fluidsim.base.solvers.pseudo_spect

Provides:

.. autoclass:: InfoSolverPseudoSpectral
   :members:
   :private-members:

.. autoclass:: SimulBasePseudoSpectral
   :members:
   :private-members:

"""

import numpy as np

from fluiddyn.util import mpi

from fluidsim.operators.setofvariables import SetOfVariables

from fluidsim.base.solvers.base import SimulBase, InfoSolverBase


class InfoSolverPseudoSpectral(InfoSolverBase):
    """Contain the information on a solver."""

    def _init_root(self):

        super(InfoSolverPseudoSpectral, self)._init_root()

        self.module_name = 'fluidsim.base.solvers.pseudo_spect'
        self.class_name = 'SimulBasePseudoSpectral'
        self.short_name = 'BasePS'

        self.classes.set_child(
            'State',
            attribs={'module_name': 'fluidsim.base.state',
                     'class_name': 'StatePseudoSpectral'})

        self.classes.set_child(
            'TimeStepping',
            attribs={'module_name':
                     'fluidsim.base.time_stepping.pseudo_spect_cy',
                     'class_name': 'TimeSteppingPseudoSpectral'})

        self.classes.set_child(
            'Operators',
            attribs={'module_name': 'fluidsim.operators.operators',
                     'class_name': 'OperatorsPseudoSpectral2D'})


info_solver_ps = InfoSolverPseudoSpectral()
info_solver_ps.complete_with_classes()


class SimulBasePseudoSpectral(SimulBase):

    @staticmethod
    def _complete_params_with_default(params):
        """A static method used to complete the *params* container."""
        SimulBase._complete_params_with_default(params)

        attribs = {'nu_8': 0.,
                   'nu_4': 0.,
                   'nu_m4': 0.}
        params.set_attribs(attribs)

    def __init__(self, params, info_solver=info_solver_ps):
        super(SimulBasePseudoSpectral, self).__init__(params, info_solver)

    def compute_freq_diss(self):
        if self.params.nu_2 > 0:
            f_d = self.params.nu_2*self.oper.K2
        else:
            f_d = np.zeros_like(self.oper.K2)

        if self.params.nu_4 > 0.:
            f_d += self.params.nu_4*self.oper.K4

        if self.params.nu_8 > 0.:
            f_d += self.params.nu_8*self.oper.K8

        if self.params.nu_m4 > 0.:
            f_d_hypo = self.params.nu_m4/self.oper.K2_not0**2
            # mode K2 = 0 !
            if mpi.rank == 0:
                f_d_hypo[0, 0] = f_d_hypo[0, 1]
        else:
            f_d_hypo = np.zeros_like(f_d)

        return f_d, f_d_hypo

    def tendencies_nonlin(self, variables=None):
        """Return a null SetOfVariables object."""
        tendencies = SetOfVariables(
            like_this_sov=self.state.state_fft,
            name_type_variables='tendencies_nonlin')
        tendencies.initialize(value=0.)
        return tendencies

Simul = SimulBasePseudoSpectral


if __name__ == "__main__":

    import fluiddyn as fld

    params = fld.simul.create_params(info_solver_ps)

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
