"""SW1L equations solving exactly the linear terms
==================================================

(:mod:`fluidsim.solvers.sw1l.onlywaves.solver`)

This class is a solver of the 1 layer shallow water (Saint Venant)
equations with zeros QG PV.
"""

import numpy as np

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.solvers.sw1l.exactlin.solver import InfoSolverSW1LExactLin
from fluidsim.solvers.sw1l.exactlin.solver import Simul as SimulSW1LExactLin


from fluiddyn.util import mpi


class InfoSolverSW1LWaves(InfoSolverSW1LExactLin):
    """Information about the solver SW1L."""

    def _init_root(self):
        super()._init_root()

        sw1l = "fluidsim.solvers.sw1l"

        self.module_name = sw1l + ".onlywaves.solver"
        self.short_name = "SW1Lwaves"

        classes = self.classes

        classes.State.module_name = sw1l + ".onlywaves.state"
        classes.State.class_name = "StateSW1LWaves"

        classes.Output.module_name = sw1l + ".onlywaves.output"
        classes.Output.class_name = "OutputSW1LWaves"


class Simul(SimulSW1LExactLin):
    """A solver of the shallow-water 1 layer equations (SW1L)"""

    InfoSolver = InfoSolverSW1LWaves

    def tendencies_nonlin(self, state_spect=None, old=None):
        oper = self.oper
        fft2 = oper.fft2

        if state_spect is None:
            state_phys = self.state.state_phys
            state_spect = self.state.state_spect
        else:
            state_phys = self.state.return_statephys_from_statespect(state_spect)

        ux = state_phys.get_var("ux")
        uy = state_phys.get_var("uy")
        eta = state_phys.get_var("eta")

        # compute the nonlinear terms for ux, uy and eta
        gradu2_x_fft, gradu2_y_fft = oper.gradfft_from_fft(
            fft2(ux**2 + uy**2) / 2
        )

        Nx_fft = -gradu2_x_fft
        Ny_fft = -gradu2_y_fft

        if self.params.f > 0:
            # not very efficient, but simple...
            rot = self.state.get_var("rot")
            N1x = +rot * uy
            N1y = -rot * ux

            Nx_fft += fft2(N1x)
            Ny_fft += fft2(N1y)

        jx_fft = fft2(eta * ux)
        jy_fft = fft2(eta * uy)
        Neta_fft = -oper.divfft_from_vecfft(jx_fft, jy_fft)

        # self.verify_tendencies(state_spect, state_phys,
        #                        Nx_fft, Ny_fft, Neta_fft)

        # compute the nonlinear terms for q, ap and am
        (Nq_fft, Np_fft, Nm_fft) = self.oper.qapamfft_from_uxuyetafft(
            Nx_fft, Ny_fft, Neta_fft
        )

        # Np_fft = self.oper.create_arrayK(value=0)
        # Nm_fft = self.oper.create_arrayK(value=0)

        oper.dealiasing(Np_fft, Nm_fft)

        if old is None:
            tendencies_fft = SetOfVariables(
                like=self.state.state_spect, info="tendencies_nonlin"
            )
        else:
            tendencies_fft = old
        tendencies_fft.set_var("ap_fft", Np_fft)
        tendencies_fft.set_var("am_fft", Nm_fft)

        if self.params.forcing.enable:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft

    def compute_freq_complex(self, key):
        K2 = self.oper.K2
        # return self.oper.create_arrayK(value=0)
        if key == "ap_fft":
            omega = 1.0j * np.sqrt(self.params.f**2 + self.params.c2 * K2)
        elif key == "am_fft":
            omega = -1.0j * np.sqrt(self.params.f**2 + self.params.c2 * K2)
        return omega

    # def verify_tendencies(
    #     self, state_spect, state_phys, Nx_fft, Ny_fft, Neta_fft
    # ):
    #     # for verification conservation energy
    #     # compute the linear terms
    #     oper = self.oper
    #     ux = state_phys.get_var("ux")
    #     uy = state_phys.get_var("uy")
    #     eta = state_phys.get_var("eta")

    #     # q_fft = self.oper.create_arrayK(value=0)
    #     ap_fft = state_spect.get_var("ap_fft")
    #     am_fft = state_spect.get_var("am_fft")
    #     a_fft = ap_fft + am_fft
    #     div_fft = self.divfft_from_apamfft(ap_fft, am_fft)

    #     eta_fft = oper.etafft_from_afft(a_fft)

    #     dx_c2eta_fft, dy_c2eta_fft = oper.gradfft_from_fft(
    #         self.params.c2 * eta_fft
    #     )
    #     LCx = self.params.f * uy
    #     LCy = -self.params.f * ux
    #     Lx_fft = oper.fft2(LCx) - dx_c2eta_fft
    #     Ly_fft = oper.fft2(LCy) - dy_c2eta_fft
    #     Leta_fft = -div_fft

    #     # compute the full tendencies
    #     Fx_fft = Lx_fft + Nx_fft
    #     Fy_fft = Ly_fft + Ny_fft
    #     Feta_fft = Leta_fft + Neta_fft
    #     oper.dealiasing(Fx_fft, Fy_fft, Feta_fft)

    #     # test : ux, uy, eta ---> q, ap, am
    #     (Fq_fft, Fp_fft, Fm_fft) = self.oper.qapamfft_from_uxuyetafft(
    #         Fx_fft, Fy_fft, Feta_fft
    #     )
    #     # test : q, ap, am ---> ux, uy, eta
    #     (Fx2_fft, Fy2_fft, Feta2_fft) = self.oper.uxuyetafft_from_qapamfft(
    #         Fq_fft, Fp_fft, Fm_fft
    #     )
    #     print(np.max(abs(Fx2_fft - Fx_fft)))
    #     print(np.max(abs(Fy2_fft - Fy_fft)))
    #     print(np.max(abs(Feta2_fft - Feta_fft)))
    #     Fx_fft = Fx2_fft
    #     Fy_fft = Fy2_fft
    #     Feta_fft = Feta2_fft

    #     (Fq2_fft, Fp2_fft, Fm2_fft) = self.oper.qapamfft_from_uxuyetafft(
    #         Fx2_fft, Fy2_fft, Feta2_fft
    #     )
    #     print(np.max(abs(Fq2_fft - Fq_fft)))
    #     print(np.max(abs(Fp2_fft - Fp_fft)))
    #     print(np.max(abs(Fm2_fft - Fm_fft)))

    #     Fx = oper.ifft2(Fx_fft)
    #     Fy = oper.ifft2(Fy_fft)
    #     Feta = oper.ifft2(Feta_fft)
    #     A = (
    #         Feta * (ux ** 2 + uy ** 2) / 2
    #         + (1 + eta) * (ux * Fx + uy * Fy)
    #         + self.params.c2 * eta * Feta
    #     )
    #     A_fft = oper.fft2(A)
    #     if mpi.rank == 0:
    #         print("should be zero =", A_fft[0, 0])


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

    sim = Simul(params)

    sim.output.phys_fields.plot()
    sim.time_stepping.start()
    sim.output.phys_fields.plot()

    fld.show()
