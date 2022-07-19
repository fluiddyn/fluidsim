"""State for the NS2D solver (:mod:`fluidsim.solvers.ns2d.state`)
=================================================================

.. autoclass:: StateNS2D
   :members:
   :private-members:

"""

import numpy as np

from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi


class StateNS2D(StatePseudoSpectral):
    """State for the solver ns2d.

    Contains the variables corresponding to the state and handles the
    access to other fields.

    .. inheritance-diagram:: StateNS2D

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""
        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": ["rot_fft"],
                "keys_state_phys": ["ux", "uy", "rot"],
                "keys_computable": [],
                "keys_phys_needed": ["rot"],
                "keys_linear_eigenmodes": ["rot_fft"],
            }
        )

    def __init__(self, sim, oper=None):

        super().__init__(sim, oper)

        self.field_tmp0 = np.empty_like(self.state_phys[0])
        self.field_tmp1 = np.empty_like(self.state_phys[0])
        self.field_tmp2 = np.empty_like(self.state_phys[0])
        self.field_tmp3 = np.empty_like(self.state_phys[0])

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        """Compute and return a variable"""
        it = self.sim.time_stepping.it
        if key in self.vars_computed and it == self.it_computed[key]:
            return self.vars_computed[key]

        if key == "ux_fft":
            # not efficient!
            result = self.oper.fft2(self.state_phys.get_var("ux"))
        elif key == "uy_fft":
            # not efficient!
            result = self.oper.fft2(self.state_phys.get_var("uy"))
        elif key == "rot_fft":
            ux_fft = self.compute("ux_fft")
            uy_fft = self.compute("uy_fft")
            result = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        elif key == "div_fft":
            ux_fft = self.compute("ux_fft")
            uy_fft = self.compute("uy_fft")
            result = self.oper.divfft_from_vecfft(ux_fft, uy_fft)
        elif key == "div":
            div_fft = self.compute("div_fft")
            result = self.oper.ifft2(div_fft)
        elif key == "q":
            rot = self.get_var("rot")
            result = rot
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

    def statephys_from_statespect(self):
        """Compute `state_phys` from `statespect`."""
        rot_fft = self.state_spect.get_var("rot_fft")
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)

        rot = self.state_phys.get_var("rot")
        ux = self.state_phys.get_var("ux")
        uy = self.state_phys.get_var("uy")

        self.oper.ifft_as_arg(rot_fft, rot)
        self.oper.ifft_as_arg(ux_fft, ux)
        self.oper.ifft_as_arg(uy_fft, uy)

    def statespect_from_statephys(self):
        """Compute `state_spect` from `state_phys`."""

        rot = self.state_phys.get_var("rot")
        rot_fft = self.state_spect.get_var("rot_fft")
        self.oper.fft_as_arg(rot, rot_fft)

    def init_from_rotfft(self, rot_fft):
        """Initialize the state from the variable `rot_fft`."""
        self.oper.dealiasing(rot_fft)
        self.state_spect.set_var("rot_fft", rot_fft)
        self.statephys_from_statespect()

    def init_from_uxfft(self, ux_fft):
        """Initialize the state from the variable `ux_fft`"""
        uy_fft = self.oper.create_arrayK(value=0.0)
        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        self.init_from_rotfft(rot_fft)

    def init_from_uyfft(self, uy_fft):
        """Initialize the state from the variable `uy_fft`"""
        ux_fft = self.oper.create_arrayK(value=0.0)
        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        self.init_from_rotfft(rot_fft)

    def init_statespect_from(self, **kwargs):
        """Initializes *state_spect* using arrays provided as keyword
        arguments.

        """
        if len(kwargs) == 1:
            key, arr = kwargs.popitem()
            if key == "rot_fft":
                self.init_from_rotfft(arr)
            elif key == "ux_fft":
                self.init_from_uxfft(arr)
            elif key == "uy_fft":
                self.init_from_uyfft(arr)
            else:
                super().init_statespect_from(**{key: arr})
        else:
            super().init_statespect_from(**kwargs)

    def init_statephys_from_ux(self, ux):
        ux_fft = self.oper.fft(ux)
        uy_fft = self.oper.create_arrayK(value=0.0)
        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        self.state_phys.set_var("rot", self.oper.ifft(rot_fft))

    def init_statephys_from_uy(self, uy):
        uy_fft = self.oper.fft(uy)
        ux_fft = self.oper.create_arrayK(value=0.0)
        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        self.state_phys.set_var("rot", self.oper.ifft(rot_fft))

    def init_statephys_from(self, **kwargs):
        if len(kwargs) == 1:
            key, arr = kwargs.popitem()
            if key == "ux":
                self.init_statephys_from_ux(arr)
            elif key == "uy":
                self.init_statephys_from_uy(arr)
            else:
                super().init_statephys_from(**{key: arr})
        else:
            return super().init_statephys_from(**kwargs)

    def compute_energy_phys(self):
        vx = self.state_phys.get_var("ux")
        vy = self.state_phys.get_var("uy")
        return 0.5 * self.sim.oper.mean_space(vx**2 + vy**2)
