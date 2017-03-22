"""Modified SW1L equations solving exactly the linear terms
===========================================================

(:mod:`fluidsim.solvers.sw1l.exactlin.modified.solver`)



"""

from __future__ import division, print_function

import numpy as np

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.solvers.sw1l.solver import InfoSolverSW1L
from fluidsim.solvers.sw1l.exactlin.solver import Simul as SimulSW1LExactLin


class InfoSolverSW1LExactLinModified(InfoSolverSW1L):
    """Information about the solver SW1L."""
    def _init_root(self):
        super(InfoSolverSW1LExactLinModified, self)._init_root()

        sw1l = 'fluidsim.solvers.sw1l'

        self.module_name = sw1l + '.exactlin.modified.solver'
        self.class_name = 'Simul'
        self.short_name = 'SW1Lexmod'

        classes = self.classes

        classes.State.module_name = sw1l + '.exactlin.state'
        classes.State.class_name = 'StateSW1LExactLin'

        classes.Output.module_name = sw1l + '.exactlin.modified.output'
        classes.Output.class_name = 'OutputSW1LExactlinModified'


class Simul(SimulSW1LExactLin):
    """A solver of the shallow-water 1 layer equations (SW1L)"""
    InfoSolver = InfoSolverSW1LExactLinModified

    def tendencies_nonlin(self, state_fft=None):
        oper = self.oper
        fft2 = oper.fft2
        ifft2 = oper.ifft2

        if state_fft is None:
            state_phys = self.state.state_phys
            state_fft = self.state.state_fft
        else:
            state_phys = self.state.return_statephys_from_statefft(state_fft)

        # compute the nonlinear terms for ux, uy and eta
        ux = state_phys.get_var('ux')
        uy = state_phys.get_var('uy')
        eta = state_phys.get_var('eta')

        rot_fft = self.state.compute('rot_fft')
        ux_rot_fft, uy_rot_fft = oper.vecfft_from_rotfft(rot_fft)
        ux_rot = ifft2(ux_rot_fft)
        uy_rot = ifft2(uy_rot_fft)

        N1x_fft = fft2(ux_rot * ux)
        N2x_fft = fft2(uy_rot * ux)
        N1y_fft = fft2(ux_rot * uy)
        N2y_fft = fft2(uy_rot * uy)

        Nx_fft = -oper.divfft_from_vecfft(N1x_fft, N2x_fft)
        Ny_fft = -oper.divfft_from_vecfft(N1y_fft, N2y_fft)

        jx_fft = fft2(ux_rot * eta)
        jy_fft = fft2(uy_rot * eta)
        Neta_fft = -oper.divfft_from_vecfft(jx_fft, jy_fft)

        # self.verify_tendencies(state_fft, state_phys, Nx_fft, Ny_fft, Neta_fft)

        # compute the nonlinear terms for q, ap and am
        (Nq_fft, Np_fft, Nm_fft
         ) = oper.qapamfft_from_uxuyetafft(Nx_fft, Ny_fft, Neta_fft)

        oper.dealiasing(Nq_fft, Np_fft, Nm_fft)

        tendencies_fft = SetOfVariables(
            like=self.state.state_fft,
            info='tendencies_nonlin')
        tendencies_fft.set_var('q_fft', Nq_fft)
        tendencies_fft.set_var('ap_fft', Np_fft)
        tendencies_fft.set_var('am_fft', Nm_fft)

        if self.params.FORCING:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft

    def verify_tendencies(self, state_fft, state_phys,
                          Nx_fft, Ny_fft, Neta_fft):
        """For verifying conservation of quadratic energy."""

        oper = self.oper
        ux_fft = self.state.compute('ux_fft')
        uy_fft = self.state.compute('uy_fft')
        eta_fft = self.state.compute('eta_fft')

        oper.dealiasing(Nx_fft, Ny_fft, Neta_fft)
        T_ux = (ux_fft.conj() * Nx_fft).real
        T_uy = (uy_fft.conj() * Ny_fft).real
        T_eta = (eta_fft.conj() * Neta_fft).real * self.params.c2
        T_tot = T_ux + T_uy + T_eta
        print('sum(T_tot) = {0:9.4e} ; sum(abs(T_tot)) = {1:9.4e}'.format(
            self.oper.sum_wavenumbers(T_tot),
            self.oper.sum_wavenumbers(abs(T_tot))))


if __name__ == "__main__":

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    nh = 64
    Lh = 2 * np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    delta_x = params.oper.Lx / params.oper.nx
    params.nu_8 = 2. * 10e-1 * params.forcing.forcing_rate ** (1. / 3) * delta_x ** 8

    params.time_stepping.t_end = 2.

    params.init_fields.type = 'vortex_grid'

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

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
