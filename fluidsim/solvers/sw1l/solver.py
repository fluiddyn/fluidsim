"""Solver one-layer shallow-water (Saint Venant) equations.
===========================================================

"""
from __future__ import division

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral, InfoSolverPseudoSpectral)

from fluiddyn.util import mpi

from .util_pythran import compute_Frot


class InfoSolverSW1L(InfoSolverPseudoSpectral):
    """Information about the solver SW1L."""
    def _init_root(self):
        super(InfoSolverSW1L, self)._init_root()

        package = 'fluidsim.solvers.sw1l'

        self.module_name = package + '.solver'
        self.class_name = 'Simul'
        self.short_name = 'SW1L'

        classes = self.classes

        classes.State.module_name = package + '.state'
        classes.State.class_name = 'StateSW1L'

        classes.InitFields.module_name = package + '.init_fields'
        classes.InitFields.class_name = 'InitFieldsSW1L'

        classes.Output.module_name = package + '.output'
        classes.Output.class_name = 'OutputSW1L'

        classes.Forcing.module_name = package + '.forcing'
        classes.Forcing.class_name = 'ForcingSW1L'


class Simul(SimulBasePseudoSpectral):
    """A solver of the shallow-water 1 layer equations (SW1L)"""

    InfoSolver = InfoSolverSW1L

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        SimulBasePseudoSpectral._complete_params_with_default(params)

        attribs = {'f': 0,
                   'c2': 20,
                   'kd2': 0,
                   'beta': 0}
        params._set_attribs(attribs)

    def __init__(self, params):
        # Parameter(s) specific to this solver
        params.kd2 = params.f**2 / params.c2

        super(Simul, self).__init__(params)

        if mpi.rank == 0:
            self.output.print_stdout(
                'c2 = {0:6.5g} ; f = {1:6.5g} ; kd2 = {2:6.5g}'.format(
                    params.c2, params.f, params.kd2))

    def tendencies_nonlin(self, state_fft=None):
        oper = self.oper

        if state_fft is None:
            state_phys = self.state.state_phys
        else:
            state_phys = self.state.return_statephys_from_statefft(state_fft)

        ux = state_phys.get_var('ux')
        uy = state_phys.get_var('uy')
        eta = state_phys.get_var('eta')
        rot = state_phys.get_var('rot')

        tendencies_fft = SetOfVariables(
            like=self.state.state_fft,
            info='tendencies_nonlin')

        Fx_fft = tendencies_fft.get_var('ux_fft')
        Fy_fft = tendencies_fft.get_var('uy_fft')
        Feta_fft = tendencies_fft.get_var('eta_fft')

        compute_tendencies_nonlin_sw1l(
            rot, ux, uy, eta,
            Fx_fft, Fy_fft, Feta_fft,
            self.params.f, self.params.beta, self.params.c2,
            oper.YY,
            oper.fft2, oper.gradfft_from_fft, oper.dealiasing,
            oper.divfft_from_vecfft)

        oper.dealiasing(tendencies_fft)

        if self.params.FORCING:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


# pythran export compute_tendencies_nonlin_sw1l(
#     float64[][], float64[][], float64[][], float64[][],
#     complex128[][], complex128[][], complex128[][],
#     float, float, float, float64[][],
#     function_to_be_called_from_python_interpreter -> complex128[][],
#     function_to_be_called_from_python_interpreter -> (
#         complex128[][], complex128[][]),
#     function_to_be_called_from_python_interpreter -> NoneType,
#     function_to_be_called_from_python_interpreter -> (
#         complex128[][], complex128[][]))

def compute_tendencies_nonlin_sw1l(
        rot, ux, uy, eta, Fx_fft, Fy_fft, Feta_fft,
        f, beta, c2, YY,
        fft2, gradfft_from_fft, dealiasing, divfft_from_vecfft):
    """Compute nonlinear tendencies for the sw1l model"""
    F1x, F1y = compute_Frot(rot, ux, uy, f, beta, YY)
    gradx_fft, grady_fft = gradfft_from_fft(
        fft2(c2 * eta + (ux**2 + uy ** 2) / 2.))
    dealiasing(gradx_fft, grady_fft)
    Fx_fft[:] = fft2(F1x) - gradx_fft
    Fy_fft[:] = fft2(F1y) - grady_fft

    Feta_fft[:] = -divfft_from_vecfft(fft2((eta + 1) * ux),
                                      fft2((eta + 1) * uy))


if __name__ == "__main__":

    import numpy as np

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    nh = 32
    Lh = 2 * np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    delta_x = params.oper.Lx / params.oper.nx
    params.nu_8 = 2. * 10e-1 * params.forcing.forcing_rate**(1. / 3) * delta_x**8

    params.time_stepping.t_end = 1.
    # params.time_stepping.USE_CFL = False
    # params.time_stepping.deltat0 = 0.01

    params.init_fields.type = 'noise'

    params.FORCING = True
    params.forcing.type = 'waves'

    params.output.periods_print.print_stdout = 0.25

    params.output.periods_save.phys_fields = 0.5
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.periods_save.increments = 0.5
    params.output.periods_save.pdf = 0.5
    params.output.periods_save.time_signals_fft = True

    params.output.periods_plot.phys_fields = 0.

    params.output.phys_fields.field_to_plot = 'eta'

    sim = Simul(params)

    sim.output.phys_fields.plot(key_field='eta')
    sim.time_stepping.start()
    sim.output.phys_fields.plot(key_field='eta')

    fld.show()
