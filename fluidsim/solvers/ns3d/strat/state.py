"""State stratified NS3D solver (:mod:`fluidsim.solvers.ns3d.strat.state`)
==========================================================================
"""

import numpy as np

from ..state import StateNS3D


class StateNS3DStrat(StateNS3D):
    """Contains the variables corresponding to the state and handles the
    access to other fields for the solver NS3D.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        keys_state_phys = ["vx", "vy", "vz", "b"]

        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": [k + "_fft" for k in keys_state_phys],
                "keys_state_phys": keys_state_phys,
                "keys_phys_needed": keys_state_phys,
                "keys_computable": ["rotz", "divh"],
                "keys_linear_eigenmodes": ["rot_fft"],
            }
        )

    def init_from_vxvyfft(self, vx_fft, vy_fft):
        self.state_spect.set_var("vx_fft", vx_fft)
        self.state_spect.set_var("vy_fft", vy_fft)
        self.state_spect.set_var("vz_fft", np.zeros_like(vx_fft))
        self.state_spect.set_var("b_fft", np.zeros_like(vx_fft))

        vx = self.oper.ifft3d(vx_fft)
        self.state_phys.set_var("vx", vx)
        self.state_phys.set_var("vy", self.oper.ifft3d(vy_fft))
        self.state_phys.set_var("vz", np.zeros_like(vx))
        self.state_phys.set_var("b", np.zeros_like(vx))

    def init_from_vxvyvzfft(self, vx_fft, vy_fft, vz_fft):
        self.state_spect.set_var("vx_fft", vx_fft)
        self.state_spect.set_var("vy_fft", vy_fft)
        self.state_spect.set_var("vz_fft", vz_fft)
        self.state_spect.set_var("b_fft", np.zeros_like(vx_fft))

        self.statephys_from_statespect()
        self.statespect_from_statephys()
