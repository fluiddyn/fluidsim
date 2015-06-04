"""Modified SW1L equations
==========================

(:mod:`fluidsim.solvers.sw1l.modified.solver`)

This class is a solver of a modified version of the 1 layer shallow
water (Saint Venant) equations for which the advection is only
due to the rotational velocity.
"""

from __future__ import division, print_function

import numpy as np

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.solvers.sw1l.solver import InfoSolverSW1L
from fluidsim.solvers.sw1l.solver import Simul as SimulSW1L


class InfoSolverSW1LModified(InfoSolverSW1L):
    """Information about the solver SW1L."""
    def _init_root(self, **kargs):
        super(InfoSolverSW1LModified, self)._init_root()

        sw1l = 'fluidsim.solvers.sw1l'

        self.module_name = sw1l+'.modified.solver'
        self.class_name = 'Simul'
        self.short_name = 'SW1Lmodif'

        classes = self.classes

        classes.State.module_name = sw1l+'.modified.state'
        classes.State.class_name = 'StateSW1LModified'

        classes.Output.module_name = sw1l+'.modified.output'
        classes.Output.class_name = 'OutputSW1LModified'


class Simul(SimulSW1L):
    """A solver of the shallow-water 1 layer equations (SW1L)"""

    InfoSolver = InfoSolverSW1LModified

    def tendencies_nonlin(self, state_fft=None):
        oper = self.oper
        fft2 = oper.fft2
        ifft2 = oper.ifft2

        if state_fft is None:
            state_phys = self.state.state_phys
            state_fft = self.state.state_fft
        else:
            state_phys = self.state.return_statephys_from_statefft(state_fft)

        ux = state_phys.get_var('ux')
        uy = state_phys.get_var('uy')
        # eta = state_phys.get_var('eta')

        ux_fft = state_fft.get_var('ux_fft')
        uy_fft = state_fft.get_var('uy_fft')
        eta_fft = state_fft.get_var('eta_fft')

        # compute Fx_fft and Fy_fft
        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        ux_rot_fft, uy_rot_fft = oper.vecfft_from_rotfft(rot_fft)
        ux_rot = ifft2(ux_rot_fft)
        uy_rot = ifft2(uy_rot_fft)

        dxux_fft, dyux_fft = oper.gradfft_from_fft(ux_fft)
        dxux = ifft2(dxux_fft)
        dyux = ifft2(dyux_fft)
        dxuy_fft, dyuy_fft = oper.gradfft_from_fft(uy_fft)
        dxuy = ifft2(dxuy_fft)
        dyuy = ifft2(dyuy_fft)

        FNLx = -ux_rot*dxux - uy_rot*dyux
        FNLy = -ux_rot*dxuy - uy_rot*dyuy

        FCx = +self.params.f*uy
        FCy = -self.params.f*ux

        Fgradx_fft, Fgrady_fft = oper.gradfft_from_fft(self.params.c2*eta_fft)

        Fx_fft = fft2(FCx+FNLx) - Fgradx_fft
        Fy_fft = fft2(FCy+FNLy) - Fgrady_fft

        # compute Feta_fft
        dxeta_fft, dyeta_fft = oper.gradfft_from_fft(eta_fft)
        dxeta = ifft2(dxeta_fft)
        dyeta = ifft2(dyeta_fft)

        div_fft = oper.divfft_from_vecfft(ux_fft, uy_fft)
        Feta_fft = -fft2(ux_rot*dxeta + uy_rot*dyeta) - div_fft

        # # for verification conservation energy
        # oper.dealiasing(Fx_fft, Fy_fft, Feta_fft)
        # T_ux = (ux_fft.conj()*Fx_fft).real
        # T_uy = (uy_fft.conj()*Fy_fft).real
        # T_eta = (eta_fft.conj()*Feta_fft).real
        # T_tot = T_ux + T_uy + T_eta
        # print 'sum(T_tot) = {0:9.4e} ; sum(abs(T_tot)) = {1:9.4e}'.format(
        #     self.oper.sum_wavenumbers(T_tot),
        #     self.oper.sum_wavenumbers(abs(T_tot)))

        tendencies_fft = SetOfVariables(
            like=self.state.state_fft,
            info='tendencies_nonlin')

        tendencies_fft.set_var('ux_fft', Fx_fft)
        tendencies_fft.set_var('uy_fft', Fy_fft)
        tendencies_fft.set_var('eta_fft', Feta_fft)

        oper.dealiasing(tendencies_fft)

        if self.params.FORCING:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


if __name__ == "__main__":

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    nh = 64
    Lh = 2*np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    delta_x = params.oper.Lx/params.oper.nx
    params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8

    params.time_stepping.t_end = 2.

    params.init_fields.type = 'noise'

    params.FORCING = True
    params.forcing.type = 'waves'

    params.output.periods_print.print_stdout = 0.25

    params.output.periods_save.phys_fields = 1.
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.periods_save.increments = 0.5
    params.output.periods_save.pdf = 0.5
    params.output.periods_save.time_signals_fft = False

    params.output.periods_plot.phys_fields = 0.

    params.output.phys_fields.field_to_plot = 'div'

    # params.output.spectra.HAS_TO_PLOT_SAVED = True
    # params.output.spatial_means.HAS_TO_PLOT_SAVED = True
    # params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True
    # params.output.increments.HAS_TO_PLOT_SAVED = True

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
