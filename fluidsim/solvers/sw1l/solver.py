"""Solver one-layer shallow-water (Saint Venant) equations.
===========================================================

"""

from fluidsim.operators.setofvariables import SetOfVariables
from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral, InfoSolverPseudoSpectral)

from fluiddyn.util import mpi


class InfoSolverSW1l(InfoSolverPseudoSpectral):
    """Information about the solver SW1l."""
    def __init__(self, **kargs):
        super(InfoSolverSW1l, self).__init__(**kargs)

        if 'tag' in kargs and kargs['tag'] == 'solver':

            package = 'fluidsim.solvers.sw1l'

            self.module_name = package + '.solver'
            self.class_name = 'Simul'
            self.short_name = 'SW1l'

            classes = self.classes

            classes.State.module_name = package + '.state'
            classes.State.class_name = 'StateSW1l'

            classes.InitFields.module_name = package + '.init_fields'
            classes.InitFields.class_name = 'InitFieldsSW1l'

            classes.Output.module_name = package + '.output'
            classes.Output.class_name = 'OutputSW1l'

            classes.Forcing.module_name = package + '.forcing'
            classes.Forcing.class_name = 'ForcingSW1l'


info_solver = InfoSolverSW1l(tag='solver')
info_solver.complete_with_classes()


class Simul(SimulBasePseudoSpectral):
    """A solver of the shallow-water 1 layer equations (SW1l)"""

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        SimulBasePseudoSpectral._complete_params_with_default(params)

        attribs = {'f': 0,
                   'c2': 20,
                   'kd2': 0}
        params.set_attribs(attribs)

    def __init__(self, params, info_solver=info_solver):
        # Parameter(s) specific to this solver
        params.kd2 = params.f**2/params.c2

        # first, the common initialisations
        super(Simul, self).__init__(params, info_solver)

        if mpi.rank == 0:
            self.output.print_stdout(
                'c2 = {0:6.5g} ; f = {1:6.5g} ; kd2 = {2:6.5g}'.format(
                    params.c2, params.f, params.kd2))

    def tendencies_nonlin(self, state_fft=None):
        oper = self.oper
        fft2 = oper.fft2

        if state_fft is None:
            state_phys = self.state.state_phys
        else:
            state_phys = self.state.return_statephys_from_statefft(state_fft)

        ux = state_phys['ux']
        uy = state_phys['uy']
        eta = state_phys['eta']
        rot = state_phys['rot']
        rot_abs = rot + self.params.f

        F1x = +rot_abs*uy
        F1y = -rot_abs*ux
        gradx_fft, grady_fft = oper.gradfft_from_fft(
            fft2(self.params.c2*eta + (ux**2+uy**2)/2))
        oper.dealiasing(gradx_fft, grady_fft)
        Fx_fft = fft2(F1x) - gradx_fft
        Fy_fft = fft2(F1y) - grady_fft

        Feta_fft = -oper.divfft_from_vecfft(fft2((eta+1)*ux),
                                            fft2((eta+1)*uy))
        oper.dealiasing(Fx_fft, Fy_fft, Feta_fft)

        # # for verification conservation energy
        # Fx   = oper.ifft2(Fx_fft)
        # Fy   = oper.ifft2(Fy_fft)
        # Feta = oper.ifft2(Feta_fft)
        # A = ( Feta*(ux**2+uy**2)/2
        #       + (1+eta)*(ux*Fx+uy*Fy)
        #       + self.params.c2*eta*Feta )
        # A_fft = fft2(A)
        # if mpi.rank == 0:
        #     print 'should be zero =', A_fft[0,0]

        tendencies_fft = SetOfVariables(
            like_this_sov=self.state.state_fft,
            name_type_variables='tendencies_nonlin'
            )
        tendencies_fft['ux_fft'] = Fx_fft
        tendencies_fft['uy_fft'] = Fy_fft
        tendencies_fft['eta_fft'] = Feta_fft

        if self.params.FORCING:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


if __name__ == "__main__":

    import numpy as np

    import fluiddyn as fld

    params = fld.simul.create_params(info_solver)

    params.short_name_type_run = 'test'

    nh = 32
    Lh = 2*np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    delta_x = params.oper.Lx/params.oper.nx
    params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8

    params.time_stepping.t_end = 2.

    params.init_fields.type_flow_init = 'NOISE'

    params.FORCING = False
    params.forcing.type = 'Random'

    params.output.periods_print.print_stdout = 0.25

    params.output.periods_save.phys_fields = 0.5
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.periods_save.increments = 0.5
    params.output.periods_save.pdf = 0.5
    params.output.periods_save.time_signals_fft = True

    params.output.periods_plot.phys_fields = 0.

    params.output.phys_fields.field_to_plot = 'rot'

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
