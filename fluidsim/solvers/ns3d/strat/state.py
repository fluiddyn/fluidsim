"""State stratified NS3D solver (:mod:`fluidsim.solvers.ns3d.strat.state`)
==========================================================================
"""

import numpy as np

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.state import StateNS3D


class StateNS3DStrat(StateNS3D):
    """State for the solver ns3d.strat.

    Contains the variables corresponding to the state and handles the
    access to other fields.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        keys_state_phys = ["vx", "vy", "vz", "b", "vp", "va"]

        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": [k + "_fft" for k in keys_state_phys],
                "keys_state_phys": keys_state_phys,
                "keys_phys_needed": keys_state_phys,
                "keys_computable": ["rotz", "divh"],
                "keys_linear_eigenmodes": ["rot_fft"],
            }
        )

    def __init__(self, sim, oper=None):
        super().__init__(sim, oper)

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        """Compute and return a variable"""
        it = self.sim.time_stepping.it
        if key in self.vars_computed and it == self.it_computed[key]:
            return self.vars_computed[key]

        if key == "rotz_fft":
            vx_fft = self.get_var("vx_fft")
            vy_fft = self.get_var("vy_fft")
            result = self.oper.rotzfft_from_vxvyfft(vx_fft, vy_fft)
        elif key == "rotz":
            rotz_fft = div_fft = self.compute("rotz_fft")
            result = self.oper.ifft3d(rotz_fft)
        elif key == "rot_fft":
            vx_fft = self.state_spect.get_var("vx_fft")
            vy_fft = self.state_spect.get_var("vy_fft")
            vz_fft = self.state_spect.get_var("vz_fft")
            result = self.oper.rotfft_from_vecfft(vx_fft, vy_fft, vz_fft)
        elif key == "divh_fft":
            vx_fft = self.get_var("vx_fft")
            vy_fft = self.get_var("vy_fft")
            result = self.oper.divhfft_from_vxvyfft(vx_fft, vy_fft)
        elif key == "divh":
            divh_fft = self.compute("divh_fft")
            result = self.oper.ifft3d(divh_fft)
        elif key == "div_fft":
            vx_fft = self.state_spect.get_var("vx_fft")
            vy_fft = self.state_spect.get_var("vy_fft")
            vz_fft = self.state_spect.get_var("vz_fft")
            result = self.oper.divfft_from_vecfft(vx_fft, vy_fft, vz_fft)
        elif key == "div":
            div_fft = self.compute("div_fft")
            result = self.oper.afft(div_fft)
        elif key == "vp_fft":
            vx_fft = self.get_var("vx_fft")
            vy_fft = self.get_var("vy_fft")
            vz_fft = self.get_var("vz_fft")
            result = self.oper.project_polar3d_scalar(vx_fft, vy_fft, vz_fft)
        elif key == "vp":
            vp_fft = self.compute("vp_fft")
            result = self.oper.ifft3d(vp_fft)
        elif key == "va_fft":
            vx_fft = self.get_var("vx_fft")
            vy_fft = self.get_var("vy_fft")
            vz_fft = self.get_var("vz_fft")
            result = self.oper.project_azim3d_scalar(vx_fft, vy_fft, vz_fft)
        elif key == "va":
            va_fft = self.compute("va_fft")
            result = self.oper.ifft3d(va_fft)
        # Remark: Implement the computation of the eigein modes a+ a- a0 and shear modes.
        else:
            to_print = 'Do not know how to compute "' + key + '".'
            if RAISE_ERROR:
                raise ValueError(to_print)

            else:
                if mpi.rank == 0:
                    print(to_print + "\nreturn an array of zeros.")

                result = self.oper.create_arrayX(value=0.0)

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

        return result

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

        self.statephys_from_statespect()
        self.statespect_from_statephys()

    def init_from_vxvyvzfft(self, vx_fft, vy_fft, vz_fft):
        self.state_spect.set_var("vx_fft", vx_fft)
        self.state_spect.set_var("vy_fft", vy_fft)
        self.state_spect.set_var("vz_fft", vz_fft)
        self.state_spect.set_var("b_fft", np.zeros_like(vx_fft))

        self.statephys_from_statespect()
        self.statespect_from_statephys()

    def init_statespect_from(self, **kwargs):
        super().init_statespect_from(**kwargs)
        # Remark: Should be changed if we want to init with the eigein modes a+ a- a0 and shear modes.
