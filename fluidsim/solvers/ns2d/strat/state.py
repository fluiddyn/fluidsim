"""State for the NS2D solver (:mod:`fluidsim.solvers.ns2d.strat.state`)
=======================================================================

.. autoclass:: StateNS2DStrat
   :members:
   :private-members:

"""

import numpy as np

# from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi

from fluidsim.solvers.ns2d.state import StateNS2D


class StateNS2DStrat(StateNS2D):
    """State for the solver ns2d.strat.

    Contains the variables corresponding to the state and handles the
    access to other fields.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Update `info_solver` container with the stratification terms."""
        # Updating the state to a stratified state
        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": ["rot_fft", "b_fft"],
                "keys_state_phys": ["ux", "uy", "rot", "b"],
                "keys_computable": [
                    "ux_fft",
                    "uy_fft",
                    "div_fft",
                    "div",
                    "ap_fft",
                    "am_fft",
                    "ap",
                    "am",
                ],
                "keys_phys_needed": ["rot", "b"],
                "keys_linear_eigenmodes": ["rot_fft", "b_fft"],
            }
        )

    def __init__(self, sim, oper=None):

        super().__init__(sim, oper)

        self.field_tmp4 = np.empty_like(self.state_phys[0])
        self.field_tmp5 = np.empty_like(self.state_phys[0])

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        """Compute and return a variable"""
        it = self.sim.time_stepping.it
        if key in self.vars_computed and it == self.it_computed[key]:
            return self.vars_computed[key]

        if key == "ux_fft":
            rot_fft = self.state_spect.get_var("rot_fft")
            ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
            result = ux_fft
        # result = self.oper.fft2(self.state_phys.get_var("ux"))
        elif key == "uy_fft":
            rot_fft = self.state_spect.get_var("rot_fft")
            ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
            result = uy_fft
        # result = self.oper.fft2(self.state_phys.get_var("uy"))
        elif key == "rot_fft":
            ux_fft = self.compute("ux_fft")
            uy_fft = self.compute("uy_fft")
            result = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        elif key == "q":
            rot = self.get_var("rot")
            result = rot
        elif key == "div_fft":
            ux_fft = self.compute("ux_fft")
            uy_fft = self.compute("uy_fft")
            result = self.oper.divfft_from_vecfft(ux_fft, uy_fft)
        elif key == "div":
            div_fft = self.compute("div_fft")
            result = self.oper.ifft2(div_fft)
        elif key == "ap_fft":
            rot_fft = self.state_spect.get_var("rot_fft")
            ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
            # print("ux_fft ap fft", ux_fft)
            # uy_fft = self.oper.fft2(self.state_phys.get_var("uy"))
            b_fft = self.state_spect.get_var("b_fft")
            N = self.sim.params.N
            omega_k = self.sim.compute_dispersion_relation()
            result = (N**2) * uy_fft + 1j * omega_k * b_fft
            # result = (N ** 2) * uy_fft - 1j * omega_k * b_fft
        elif key == "am_fft":
            rot_fft = self.state_spect.get_var("rot_fft")
            # print("rot_fft", rot_fft)
            ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
            # print("ux_fft am fft", ux_fft)
            # print("uy_fft", uy_fft)
            # uy_fft = self.oper.fft2(self.state_phys.get_var("uy"))
            b_fft = self.state_spect.get_var("b_fft")
            # print("b_fft", b_fft)
            N = self.sim.params.N
            omega_k = self.sim.compute_dispersion_relation()
            # print("omega_k", omega_k)
            # print("1j * omega_k * b_fft", 1j * omega_k * b_fft)
            # print("(N ** 2) * uy_fft", (N ** 2) * uy_fft)
            result = (N**2) * uy_fft - 1j * omega_k * b_fft
            # result = (N ** 2) * uy_fft + 1j * omega_k * b_fft
        # result = np.zeros(self.oper.shapeK, dtype="complex")
        elif key == "ap":
            ap_fft = self.compute("ap_fft")
            result = self.oper.ifft(ap_fft)
        elif key == "am":
            am_fft = self.compute("am_fft")
            result = self.oper.ifft(am_fft)

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
        b_fft = self.state_spect.get_var("b_fft")
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)

        rot = self.state_phys.get_var("rot")
        ux = self.state_phys.get_var("ux")
        uy = self.state_phys.get_var("uy")
        b = self.state_phys.get_var("b")

        self.oper.ifft_as_arg(rot_fft, rot)
        self.oper.ifft_as_arg(ux_fft, ux)
        self.oper.ifft_as_arg(uy_fft, uy)
        self.oper.ifft_as_arg(b_fft, b)

    def statespect_from_statephys(self):
        """Compute `state_spect` from `state_phys`."""

        rot = self.state_phys.get_var("rot")
        b = self.state_phys.get_var("b")

        rot_fft = self.state_spect.get_var("rot_fft")
        b_fft = self.state_spect.get_var("b_fft")

        self.oper.fft_as_arg(rot, rot_fft)
        self.oper.fft_as_arg(b, b_fft)

    def init_from_rotbfft(self, rot_fft, b_fft):
        """Initialize the state from the variable rot_fft and b_fft."""
        self.oper.dealiasing(rot_fft)
        self.oper.dealiasing(b_fft)

        self.state_spect.set_var("rot_fft", rot_fft)
        self.state_spect.set_var("b_fft", b_fft)

        self.statephys_from_statespect()

    def init_from_apfft(self, ap_fft):
        """Initialize the state from the linear mode ap_fft."""
        rot_fft, b_fft = self.compute_rotbfft_from_apfft(ap_fft)
        self.init_from_rotbfft(rot_fft, b_fft)

    def init_from_amfft(self, am_fft):
        """Initialize the state from the linear mode am_fft."""
        rot_fft, b_fft = self.compute_rotbfft_from_amfft(am_fft)
        self.init_from_rotbfft(rot_fft, b_fft)

    def compute_rotbfft_from_amfft(self, am_fft):
        """
        Computes rot_fft and b_fft from linear mode am_fft.

        ap_fft is assumed to be null
        """
        # it should be (ap_fft + am_fft)
        # but ap_fft is assumed to be null
        uy_fft = (1.0 / (2 * self.params.N**2)) * am_fft
        cond = self.oper.KX != 0
        division = np.zeros_like(self.oper.KY)
        division[cond] = self.oper.KY[cond] / self.oper.KX[cond]
        ux_fft = -1 * division * uy_fft

        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)

        omega_k = self.params.N * self.oper.KX / self.oper.K_not0
        b_fft = np.zeros(self.oper.shapeK, dtype="complex")

        # it should be (ap_fft[cond] - am_fft[cond])
        # but ap_fft is assumed to be null
        b_fft[cond] = (1.0 / (2j * omega_k[cond])) * (-am_fft[cond])
        return rot_fft, b_fft

    def compute_rotbfft_from_apfft(self, ap_fft):
        """
        Computes rot_fft and b_fft from linear mode ap_fft.

        am_fft is assumed to be null
        """
        # it should be (ap_fft + am_fft)
        # but am_fft is assumed to be null
        uy_fft = (1.0 / (2 * self.params.N**2)) * ap_fft
        cond = self.oper.KX != 0
        division = np.zeros_like(self.oper.KY)
        division[cond] = self.oper.KY[cond] / self.oper.KX[cond]
        ux_fft = -1 * division * uy_fft

        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)

        omega_k = self.params.N * self.oper.KX / self.oper.K_not0
        b_fft = np.zeros(self.oper.shapeK, dtype="complex")
        # it should be (ap_fft[cond] - am_fft[cond])
        # but am_fft is assumed to be null
        b_fft[cond] = (1.0 / (2j * omega_k[cond])) * ap_fft[cond]

        return rot_fft, b_fft

    def init_from_rotfft(self, rot_fft):
        """Initialize the state from the variable `rot_fft`."""
        b_fft = self.oper.create_arrayK(value=0.0)
        self.init_from_rotbfft(rot_fft, b_fft)

    def init_statespect_from(self, **kwargs):
        if len(kwargs) == 1:
            key, arr = kwargs.popitem()
            if key == "rot_fft":
                self.init_from_rotfft(arr)
            elif key == "ap_fft":
                self.init_from_apfft(arr)
            elif key == "am_fft":
                self.init_from_amfft(arr)
            elif key == "ux_fft":
                self.init_from_uxfft(arr)
            elif key == "uy_fft":
                self.init_from_uyfft(arr)
            else:
                super(StateNS2D, self).init_statespect_from(**kwargs)
        elif len(kwargs) == 2:
            if "rot_fft" in kwargs and "b_fft" in kwargs:
                self.init_from_rotbfft(kwargs["rot_fft"], kwargs["b_fft"])
            else:
                super(StateNS2D, self).init_statespect_from(**kwargs)
        else:
            super(StateNS2D, self).init_statespect_from(**kwargs)
