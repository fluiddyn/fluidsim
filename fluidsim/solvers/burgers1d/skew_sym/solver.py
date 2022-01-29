"""Burgers1D solver (:mod:`fluidsim.solvers.burgers1d.skew_sym.solver`)
=======================================================================

Provides:

.. autoclass:: Simul
   :members:
   :private-members:


"""
from fluidsim.base.setofvariables import SetOfVariables
from ..solver import Simul as SimulBurgers, InfoSolver as InfoSolverBurgers


class InfoSolver(InfoSolverBurgers):
    def _init_root(self):
        super()._init_root()

        package = "fluidsim.solvers.burgers1d.skew_sym"
        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "BurgersSkewSym1D"


class Simul(SimulBurgers):
    InfoSolver = InfoSolver

    def tendencies_nonlin(self, state_spect=None, old=None):

        if state_spect is None:
            u_fft = self.state.state_spect.get_var("u_fft")
            signal = self.state.state_phys.get_var("u")
        else:
            u_fft = state_spect.get_var("u_fft")
            signal = self.oper.ifft(u_fft)

        pxu_fft = self.oper.pxffft_from_fft(u_fft)
        pxu = self.oper.ifft(pxu_fft)

        # Half from convective form (in physical space)
        f_signal = -0.5 * signal * pxu

        # Half from conservative form (in spectral space)
        f_fft_cons = -0.25 * self.oper.pxffft_from_fft(self.oper.fft(signal**2))

        if old is None:
            tendencies_fft = SetOfVariables(
                like=self.state.state_spect, info="tendencies_nonlin"
            )
        else:
            tendencies_fft = old

        f_fft = tendencies_fft.get_var("u_fft")
        self.oper.fft_as_arg(f_signal, f_fft)

        # Add the conservative form part to yield the skew symmetric form
        f_fft[:] += f_fft_cons

        self.oper.dealiasing(f_fft)
        # Set "oddball mode" to zero
        f_fft[self.oper.nkx - 1] = 0.0
        return tendencies_fft
