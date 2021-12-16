"""State stratified NS3D solver (:mod:`fluidsim.solvers.ns3d.strat.state`)
==========================================================================
"""

import numpy as np

from fluidsim.solvers.ns3d.state import StateNS3D


class StateNS3DStrat(StateNS3D):
    """State for the solver ns3d.strat.

    Contains the variables corresponding to the state and handles the
    access to other fields.

    """

    @classmethod
    def _complete_info_solver(cls, info_solver):
        """Complete the ParamContainer info_solver."""

        super()._complete_info_solver(info_solver)

        info_State = info_solver.classes.State
        keys_state_phys = ["vx", "vy", "vz", "b"]
        info_State.keys_state_spect = [k + "_fft" for k in keys_state_phys]
        info_State.keys_state_phys = keys_state_phys
        info_State.keys_phys_needed = keys_state_phys

    def init_from_vxvyfft(self, vx_fft, vy_fft):
        self.state_spect.fill(0.0)
        self.state_spect.set_var("vx_fft", vx_fft)
        self.state_spect.set_var("vy_fft", vy_fft)

        self.statephys_from_statespect()
        self.statespect_from_statephys()

    def init_from_vxvyvzfft(self, vx_fft, vy_fft, vz_fft):
        self.state_spect.set_var("vx_fft", vx_fft)
        self.state_spect.set_var("vy_fft", vy_fft)
        self.state_spect.set_var("vz_fft", vz_fft)
        self.state_spect.set_var("b_fft", np.zeros_like(vx_fft))

        self.statephys_from_statespect()
        self.statespect_from_statephys()
