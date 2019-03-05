import numpy as np

from fluidsim.base.time_stepping.pseudo_spect import TimeSteppingPseudoSpectral


class TimeSteppingPseudoSpectralNS3D(TimeSteppingPseudoSpectral):
    def one_time_step_computation(self):
        """One time step"""
        state_spect = self.sim.state.state_spect
        # WARNING: if the function _time_step_RK comes from an extension, its
        # execution time seems to be attributed to the function
        # one_time_step_computation by cProfile
        self._time_step_RK()
        self.sim.oper.dealiasing(state_spect)
        vx_fft = state_spect.get_var("vx_fft")
        vy_fft = state_spect.get_var("vy_fft")
        vz_fft = state_spect.get_var("vz_fft")
        self.sim.oper.project_perpk3d(vx_fft, vy_fft, vz_fft)
        self.sim.state.statephys_from_statespect()
        # np.isnan(np.sum seems to be really fast
        if np.isnan(np.sum(state_spect[0])):
            raise ValueError(f"nan at it = {self.it}, t = {self.t:.4f}")
