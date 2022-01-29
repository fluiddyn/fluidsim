"""Modified SW1L equations
==========================

(:mod:`fluidsim.solvers.sw1l.modified.solver`)

This class is a solver of a modified version of the 1 layer shallow
water (Saint Venant) equations for which the advection is only
due to the rotational velocity.
"""

import numpy as np

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.solvers.sw1l.solver import InfoSolverSW1L
from fluidsim.solvers.sw1l.solver import Simul as SimulSW1L


class InfoSolverSW1LModified(InfoSolverSW1L):
    """Information about the solver SW1L."""

    def _init_root(self, **kargs):
        super()._init_root()

        sw1l = "fluidsim.solvers.sw1l"

        self.module_name = sw1l + ".modified.solver"
        self.class_name = "Simul"
        self.short_name = "SW1Lmodif"

        classes = self.classes

        classes.State.module_name = sw1l + ".modified.state"
        classes.State.class_name = "StateSW1LModified"

        classes.Output.module_name = sw1l + ".modified.output"
        classes.Output.class_name = "OutputSW1LModified"


class Simul(SimulSW1L):
    """A solver of the shallow-water 1 layer equations (SW1L)"""

    InfoSolver = InfoSolverSW1LModified

    def tendencies_nonlin(self, state_spect=None, old=None):
        oper = self.oper
        fft2 = oper.fft2
        ifft2 = oper.ifft2

        if state_spect is None:
            state_phys = self.state.state_phys
            state_spect = self.state.state_spect
        else:
            state_phys = self.state.return_statephys_from_statespect(state_spect)

        ux = state_phys.get_var("ux")
        uy = state_phys.get_var("uy")
        # eta = state_phys.get_var('eta')

        ux_fft = state_spect.get_var("ux_fft")
        uy_fft = state_spect.get_var("uy_fft")
        eta_fft = state_spect.get_var("eta_fft")

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

        FNLx = -ux_rot * dxux - uy_rot * dyux
        FNLy = -ux_rot * dxuy - uy_rot * dyuy

        if self.params.beta != 0:
            f = self.params.f + self.params.beta * oper.YY
        else:
            f = self.params.f

        FCx = +f * uy
        FCy = -f * ux

        Fgradx_fft, Fgrady_fft = oper.gradfft_from_fft(self.params.c2 * eta_fft)

        Fx_fft = fft2(FCx + FNLx) - Fgradx_fft
        Fy_fft = fft2(FCy + FNLy) - Fgrady_fft

        # compute Feta_fft
        dxeta_fft, dyeta_fft = oper.gradfft_from_fft(eta_fft)
        dxeta = ifft2(dxeta_fft)
        dyeta = ifft2(dyeta_fft)

        div_fft = oper.divfft_from_vecfft(ux_fft, uy_fft)
        Feta_fft = -fft2(ux_rot * dxeta + uy_rot * dyeta) - div_fft

        if old is None:
            tendencies_fft = SetOfVariables(
                like=self.state.state_spect, info="tendencies_nonlin"
            )
        else:
            tendencies_fft = old

        tendencies_fft.set_var("ux_fft", Fx_fft)
        tendencies_fft.set_var("uy_fft", Fy_fft)
        tendencies_fft.set_var("eta_fft", Feta_fft)

        oper.dealiasing(tendencies_fft)

        if self.params.forcing.enable:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


if __name__ == "__main__":

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    nh = 64
    Lh = 2 * np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    delta_x = params.oper.Lx / params.oper.nx
    params.nu_8 = (
        2.0 * 10e-1 * params.forcing.forcing_rate ** (1.0 / 3) * delta_x**8
    )

    params.time_stepping.t_end = 2.0

    params.init_fields.type = "noise"

    params.forcing.enable = True
    params.forcing.type = "waves"

    params.output.periods_print.print_stdout = 0.25

    params.output.periods_save.phys_fields = 1.0
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.periods_save.increments = 0.5
    params.output.periods_save.pdf = 0.5
    params.output.periods_save.time_signals_fft = False

    params.output.periods_plot.phys_fields = 0.0

    params.output.phys_fields.field_to_plot = "div"

    # params.output.spectra.HAS_TO_PLOT_SAVED = True
    # params.output.spatial_means.HAS_TO_PLOT_SAVED = True
    # params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True
    # params.output.increments.HAS_TO_PLOT_SAVED = True

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
