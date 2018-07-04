"""SW1L equations solved in vorticity-divergence formulation
============================================================

(:mod:`fluidsim.solvers.sw1l.betaplane.solver`)



"""

from __future__ import division, print_function

import numpy as np

from fluiddyn.util import mpi
from fluidsim.base.setofvariables import SetOfVariables


from fluidsim.solvers.sw1l.solver import InfoSolverSW1L
from fluidsim.solvers.sw1l.solver import Simul as SimulSW1L, SimulBasePseudoSpectral

from fluidsim.solvers.ns2d.util_pythran import compute_Frot
from fluidsim.solvers.sw1l.betaplane.util_pythran import compute_Fdiv



class InfoSolverSW1LBetaPlane(InfoSolverSW1L):
    """Information about the solver SW1L."""

    def _init_root(self):
        super(InfoSolverSW1LBetaPlane, self)._init_root()

        sw1l = "fluidsim.solvers.sw1l"

        self.module_name = sw1l + ".betaplane.solver"
        self.class_name = "Simul"
        self.short_name = "SW1Lbeta"

        classes = self.classes

        classes.State.module_name = sw1l + ".betaplane.state"
        classes.State.class_name = "StateSW1LBetaPlane"


class Simul(SimulSW1L, SimulBasePseudoSpectral):
    """A solver of the shallow-water 1 layer equations (SW1L)"""

    InfoSolver = InfoSolverSW1LBetaPlane

    def __init__(self, params):
        # Parameter(s) specific to this solver
        params.kd2 = params.f ** 2 / params.c2

        # super(Simul, self).__init__(params)
        SimulBasePseudoSpectral.__init__(self, params)

        if mpi.rank == 0:
            self.output.print_stdout(
                "c2 = {0:6.5g} ; f = {1:6.5g} ; beta = {2:6.5g} ; kd2 = {3:6.5g}".format(
                    params.c2, params.f, params.beta, params.kd2
                )
            )


    def tendencies_nonlin(self, state_spect=None, old=None):
        # the operator and the fast Fourier transform
        oper = self.oper
        ifft_as_arg = oper.ifft_as_arg

        # get or compute rot_fft, ux and uy
        if state_spect is None:
            state_spect = self.state.state_spect
            rot_fft = state_spect.get_var("rot_fft")
            div_fft = state_spect.get_var("div_fft")
            eta_fft = state_spect.get_var("eta_fft")
            ux = self.state.state_phys.get_var("ux")
            uy = self.state.state_phys.get_var("uy")
            eta = self.state.state_phys.get_var("eta")
        else:
            rot_fft = state_spect.get_var("rot_fft")
            div_fft = state_spect.get_var("div_fft")
            eta_fft = state_spect.get_var("eta_fft")
            urx_fft, ury_fft = oper.vecfft_from_rotfft(rot_fft)
            udx_fft, udy_fft = oper.vecfft_from_divfft(div_fft)
            ux_fft = urx_fft + udx_fft
            uy_fft = ury_fft + udy_fft
            if mpi.rank == 0:
                ux_fft[0, 0] = 0. + 0j  # TODO: Is it OK?
                uy_fft[0, 0] = 0. + 0j  # TODO: Is it OK?
            ux = self.state.field_tmp0
            uy = self.state.field_tmp1
            eta = self.state.field_tmp6
            ifft_as_arg(ux_fft, ux)
            ifft_as_arg(uy_fft, uy)
            ifft_as_arg(eta_fft, eta)

        # TODO: make rot and div part of state_phys
        rot = oper.ifft(rot_fft)
        div = oper.ifft(div_fft)

        # "px" like $\partial_x$
        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_rot = self.state.field_tmp2
        py_rot = self.state.field_tmp3
        ifft_as_arg(px_rot_fft, px_rot)
        ifft_as_arg(py_rot_fft, py_rot)

        px_div_fft, py_div_fft = oper.gradfft_from_fft(div_fft)
        px_div = self.state.field_tmp4
        py_div = self.state.field_tmp5
        ifft_as_arg(px_div_fft, px_div)
        ifft_as_arg(py_div_fft, py_div)

        f = float(self.params.f)
        beta = float(self.params.beta)
        Frot = compute_Frot(ux, uy, px_rot, py_rot, beta)
        Fdiv = compute_Fdiv(ux, uy, px_div, py_div, beta)
        Fdiv -= oper.ifft(self.params.c2 * oper.laplacian_fft(eta_fft))
        if f != 0:
            Frot -= f * div
            Fdiv += f * rot


        if old is None:
            tendencies_fft = SetOfVariables(like=self.state.state_spect)
        else:
            tendencies_fft = old

        Frot_fft = tendencies_fft.get_var("rot_fft")
        Fdiv_fft = tendencies_fft.get_var("div_fft")
        Feta_fft = tendencies_fft.get_var("eta_fft")      

        oper.fft_as_arg(Frot, Frot_fft)
        oper.fft_as_arg(Fdiv, Fdiv_fft)
        Feta_fft[:] = -oper.divfft_from_vecfft(
            oper.fft((eta + 1) * ux), oper.fft((eta + 1) * uy))

        oper.dealiasing(tendencies_fft)

        if self.params.forcing.enable:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


if __name__ == "__main__":

    import fluiddyn as fld
    # To be compared with ns2d solver with beta term
    # from fluidsim.solvers.ns2d.solver import Simul

    params = Simul.create_default_params()

    params.short_name_type_run = "test"
    params.output.sub_directory = "test"

    nh = 144
    Lh = 2 * np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    P = 1.
    # params.f = 0
    # k_beta = 4.8
    # params.beta = (k_beta ** 5 * P) ** (1. / 3)
    params.beta = 1.5

    # delta_x = params.oper.Lx / params.oper.nx
    #    params.nu_8 = (
    #        2. * 10e-1 * params.forcing.forcing_rate ** (1. / 3) * delta_x ** 8
    #    )

    params.time_stepping.t_end = 25.

    params.init_fields.type = "constant"
    params.init_fields.constant.value = 0.

    params.forcing.enable = True
    params.forcing.type = "tcrandom"
    params.forcing.forcing_rate = 1.
    params.forcing.nkmin_forcing = 10
    params.forcing.nkmax_forcing = 14
    params.forcing.key_forced = "q_fft"
    
    params.preprocess.enable = True
    params.preprocess.forcing_const = P
    # params.preprocess.forcing_scale = "enstrophy"
    params.preprocess.viscosity_const = 0.4
    params.preprocess.viscosity_type = "hyper8"
    params.preprocess.viscosity_scale = "forcing"


    params.output.periods_print.print_stdout = 0.5
    params.output.periods_save.spatial_means = 0.5
    params.output.periods_save.phys_fields = 1.
    params.output.periods_save.spectra = 5


    params.output.periods_plot.phys_fields = 1.
    params.output.spatial_means.HAS_TO_PLOT_SAVED = True
    params.output.spectra.HAS_TO_PLOT_SAVED = True


    params.output.phys_fields.field_to_plot = "rot"

    sim = Simul(params)
    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
