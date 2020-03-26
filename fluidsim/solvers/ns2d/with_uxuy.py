import numpy as np

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.base.time_stepping.pseudo_spect import TimeSteppingPseudoSpectral

from .state import StateNS2D as StateBase
from .solver import InfoSolverNS2D as InfoBase, Simul as SimulBase, compute_Frot
from .output import Output as OutputBase


class TimeStepping(TimeSteppingPseudoSpectral):
    def one_time_step_computation(self):
        """One time step"""
        state_spect = self.sim.state.state_spect
        self._time_step_RK()
        self.sim.oper.dealiasing(state_spect)
        ux_fft = state_spect.get_var("ux_fft")
        uy_fft = state_spect.get_var("uy_fft")
        self.sim.oper.projection_perp(ux_fft, uy_fft)
        self.sim.state.statephys_from_statespect()
        if np.isnan(np.sum(state_spect[0])):
            raise ValueError(f"nan at it = {self.it}, t = {self.t:.4f}")


class State(StateBase):
    @staticmethod
    def _complete_info_solver(info_solver):
        StateBase._complete_info_solver(info_solver)
        info_solver.classes.State.keys_state_spect = ["ux_fft", "uy_fft"]

    def init_statespect_from(self, **kwargs):
        if len(kwargs) == 2 and sorted(kwargs.keys()) == ["ux_fft", "uy_fft"]:
            self.state_spect.set_var("ux_fft", kwargs["ux_fft"])
            self.state_spect.set_var("uy_fft", kwargs["uy_fft"])
            self.statephys_from_statespect()
            return
        super().init_statespect_from(**kwargs)

    def init_from_rotfft(self, rot_fft):
        ux_fft, uy_fft = self.sim.oper.vecfft_from_rotfft(rot_fft)
        self.state_spect.set_var("ux_fft", ux_fft)
        self.state_spect.set_var("uy_fft", uy_fft)
        self.statephys_from_statespect()

    def statephys_from_statespect(self):
        ux_fft = self.state_spect.get_var("ux_fft")
        uy_fft = self.state_spect.get_var("uy_fft")
        rot_fft = self.sim.oper.rotfft_from_vecfft(ux_fft, uy_fft)

        rot = self.state_phys.get_var("rot")
        ux = self.state_phys.get_var("ux")
        uy = self.state_phys.get_var("uy")

        self.oper.ifft_as_arg(rot_fft, rot)
        self.oper.ifft_as_arg(ux_fft, ux)
        self.oper.ifft_as_arg(uy_fft, uy)


class Output(OutputBase):
    def compute_energy_fft(self):
        """Compute energy(k)"""
        ux_fft = self.sim.state.state_spect.get_var("ux_fft")
        uy_fft = self.sim.state.state_spect.get_var("uy_fft")
        return (np.abs(ux_fft) ** 2 + np.abs(uy_fft) ** 2) / 2.0


class InfoSolver(InfoBase):
    def _init_root(self):

        super()._init_root()

        module = "fluidsim.solvers.ns2d.with_uxuy"
        self.module_name = module
        self.class_name = "Simul"
        self.short_name = "NS2D"

        classes = self.classes

        classes.State.module_name = module
        classes.State.class_name = "State"

        classes.Output.module_name = module
        classes.Output.class_name = "Output"

        classes.TimeStepping.module_name = module
        classes.TimeStepping.class_name = "TimeStepping"


class Simul(SimulBase):
    InfoSolver = InfoSolver

    def tendencies_nonlin(self, state_spect=None, old=None):

        # the operator and the fast Fourier transform
        oper = self.oper
        ifft_as_arg = oper.ifft_as_arg

        # get or compute rot_fft, ux and uy
        if state_spect is None:
            ux_fft = self.state.state_spect.get_var("ux_fft")
            uy_fft = self.state.state_spect.get_var("uy_fft")
            rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
            ux = self.state.state_phys.get_var("ux")
            uy = self.state.state_phys.get_var("uy")
        else:
            ux_fft = state_spect.get_var("ux_fft")
            uy_fft = state_spect.get_var("uy_fft")
            rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
            ux = self.state.field_tmp0
            uy = self.state.field_tmp1
            ifft_as_arg(ux_fft, ux)
            ifft_as_arg(uy_fft, uy)

        # "px" like $\partial_x$
        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)

        px_rot = self.state.field_tmp2
        py_rot = self.state.field_tmp3

        ifft_as_arg(px_rot_fft, px_rot)
        ifft_as_arg(py_rot_fft, py_rot)

        Frot = compute_Frot(ux, uy, px_rot, py_rot, self.params.beta)

        if old is None:
            tendencies_fft = SetOfVariables(like=self.state.state_spect)
        else:
            tendencies_fft = old

        Frot_fft = oper.fft(Frot)
        oper.dealiasing(Frot_fft)

        Fx_fft, Fy_fft = self.oper.vecfft_from_rotfft(Frot_fft)

        tendencies_fft.set_var("ux_fft", Fx_fft)
        tendencies_fft.set_var("uy_fft", Fy_fft)

        if self.params.forcing.enable:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft
