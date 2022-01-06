import numpy as np

from fluidsim.base.time_stepping.pseudo_spect import TimeSteppingPseudoSpectral
from fluidsim.operators.operators3d import dealiasing_variable


class TimeSteppingPseudoSpectralNS3D(TimeSteppingPseudoSpectral):
    def one_time_step_computation(self):
        """One time step"""
        state_spect = self.sim.state.state_spect
        # WARNING: if the function _time_step_RK comes from an extension, its
        # execution time seems to be attributed to the function
        # one_time_step_computation by cProfile
        self._time_step_RK()
        self.sim.project_state_spect(state_spect)
        self.sim.oper.dealiasing(state_spect)
        self.sim.state.statephys_from_statespect()
        # np.isnan(np.sum seems to be really fast
        if np.isnan(np.sum(state_spect[0])):
            raise ValueError(f"nan at it = {self.it}, t = {self.t:.4f}")
