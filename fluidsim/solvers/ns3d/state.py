"""State for the NS3D solver (:mod:`fluidsim.solvers.ns3d.state`)
=======================================================================
"""

import numpy as np


from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi


class StateNS3D(StatePseudoSpectral):
    """Contains the variables corresponding to the state and handles the
    access to other fields for the solver NS3D.

    """

    @classmethod
    def _complete_info_solver(cls, info_solver):
        """Complete the ParamContainer info_solver."""
        keys_state_phys = ["vx", "vy", "vz"]

        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": [k + "_fft" for k in keys_state_phys],
                "keys_state_phys": keys_state_phys,
                "keys_phys_needed": keys_state_phys,
                "keys_computable": [
                    "rotz",
                    "divh",
                    "rotz_fft",
                    "divh_fft",
                    "vp",
                    "vt",
                    "vp_fft",
                    "vt_fft",
                ],
                "keys_linear_eigenmodes": ["vp_fft", "vt_fft"],
            }
        )

    def __init__(self, sim, oper=None):

        super().__init__(sim, oper)

        self.fields_tmp = tuple(
            np.empty_like(self.state_phys[0]) for n in range(6)
        )

        self.fields_spect_tmp = tuple(
            np.empty_like(self.state_spect[0]) for n in range(3)
        )

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        it = self.sim.time_stepping.it
        if key in self.vars_computed and it == self.it_computed[key]:
            return self.vars_computed[key]

        if key == "rotz_fft":
            vx_fft = self.get_var("vx_fft")
            vy_fft = self.get_var("vy_fft")
            result = self.oper.rotzfft_from_vxvyfft(vx_fft, vy_fft)
        elif key == "rotz":
            vx_fft = self.get_var("vx_fft")
            vy_fft = self.get_var("vy_fft")
            rotz_fft = self.oper.rotzfft_from_vxvyfft(vx_fft, vy_fft)
            result = self.oper.ifft3d(rotz_fft)
        elif key == "divh":
            vx_fft = self.get_var("vx_fft")
            vy_fft = self.get_var("vy_fft")
            divh_fft = self.oper.divhfft_from_vxvyfft(vx_fft, vy_fft)
            result = self.oper.ifft3d(divh_fft)
        elif key == "divh_fft":
            vx_fft = self.get_var("vx_fft")
            vy_fft = self.get_var("vy_fft")
            result = self.oper.divhfft_from_vxvyfft(vx_fft, vy_fft)
        elif key == "vp_fft":
            vx_fft = self.get_var("vx_fft")
            vy_fft = self.get_var("vy_fft")
            vz_fft = self.get_var("vz_fft")
            result = self.oper.vpfft_from_vecfft(vx_fft, vy_fft, vz_fft)
        elif key == "vp":
            vp_fft = self.compute("vp_fft")
            result = self.oper.ifft3d(vp_fft)
        elif key == "vt_fft":
            vx_fft = self.get_var("vx_fft")
            vy_fft = self.get_var("vy_fft")
            vz_fft = self.get_var("vz_fft")
            result = self.oper.vtfft_from_vecfft(vx_fft, vy_fft, vz_fft)
        elif key == "vt":
            vt_fft = self.compute("vt_fft")
            result = self.oper.ifft3d(vt_fft)

        else:
            to_print = f'Do not know how to compute "{key}".'
            if RAISE_ERROR:
                raise ValueError(to_print)

            else:
                mpi.printby0(to_print + "\nreturn an array of zeros.")

                result = self.oper.create_arrayX(value=0.0)

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

        return result

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

        self.statephys_from_statespect()
        self.statespect_from_statephys()

    def init_statespect_from(self, **kwargs):
        """Initialize `state_spect` from arrays.

        Parameters
        ----------

        **kwargs : {key: array, ...}

          keys and arrays used for the initialization. The other keys
          are set to zero.

        Examples
        --------

        .. code-block:: python

           kwargs = {'a_fft': Fa_fft}
           init_statespect_from(**kwargs)

           ux_fft, uy_fft, eta_fft = oper.uxuyetafft_from_qfft(q_fft)
           init_statespect_from(ux_fft=ux_fft, uy_fft=uy_fft, eta_fft=eta_fft)

        """

        if len(kwargs) == 1:
            key, field = kwargs.copy().popitem()
            self.state_spect[:] = 0.0
            if key == "rotz_fft":
                vx_fft, vy_fft = self.oper.vxvyfft_from_rotzfft(field)
                return super().init_statespect_from(vx_fft=vx_fft, vy_fft=vy_fft)
            elif key == "divh_fft":
                vx_fft, vy_fft = self.oper.vxvyfft_from_divhfft(field)
                return super().init_statespect_from(vx_fft=vx_fft, vy_fft=vy_fft)
            elif key == "vp_fft":
                vx_fft, vy_fft, vz_fft = self.oper.vecfft_from_vpfft(field)
                return super().init_statespect_from(
                    vx_fft=vx_fft, vy_fft=vy_fft, vz_fft=vz_fft
                )
            elif key == "vt_fft":
                vx_fft, vy_fft, vz_fft = self.oper.vecfft_from_vtfft(field)
                return super().init_statespect_from(
                    vx_fft=vx_fft, vy_fft=vy_fft, vz_fft=vz_fft
                )
        super().init_statespect_from(**kwargs)

    def compute_energy_phys(self):
        vx = self.state_phys.get_var("vx")
        vy = self.state_phys.get_var("vy")
        vz = self.state_phys.get_var("vz")
        return 0.5 * self.sim.oper.mean_space(vx**2 + vy**2 + vz**2)
