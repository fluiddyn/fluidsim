"""State class for the SW1L.onlywaves solver
(:mod:`fluidsim.solvers.sw1l.onlywaves.state`)
====================================================


Provides:

.. autoclass:: StateSW1LWaves
   :members:
   :private-members:

"""

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.solvers.sw1l.state import StateSW1L

from fluiddyn.util import mpi


class StateSW1LWaves(StateSW1L):
    """
    The class :class:`StateSW1Lwaves` contains the variables corresponding
    to the state and handles the access to other fields for the solver
    SW1L.
    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": ["ap_fft", "am_fft"],
                "keys_state_phys": ["ux", "uy", "eta"],
                "keys_computable": [],
                "keys_phys_needed": ["ux", "uy", "eta"],
                "keys_linear_eigenmodes": ["q_fft", "a_fft", "d_fft"],
            }
        )

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        it = self.sim.time_stepping.it
        if key in self.vars_computed and it == self.it_computed[key]:
            return self.vars_computed[key]

        if key == "div_fft":
            ap_fft = self.state_spect.get_var("ap_fft")
            am_fft = self.state_spect.get_var("am_fft")
            d_fft = self.oper.divfft_from_apamfft(ap_fft, am_fft)
            result = d_fft

        elif key == "a_fft":
            ap_fft = self.state_spect.get_var("ap_fft")
            am_fft = self.state_spect.get_var("am_fft")
            result = ap_fft + am_fft

        elif key == "rot_fft":
            a_fft = self.compute("a_fft")
            result = self.oper.rotfft_from_afft(a_fft)

        elif key == "eta_fft":
            a_fft = self.compute("a_fft")
            result = self.oper.etafft_from_afft(a_fft)

        elif key == "ux_fft":
            rot_fft = self.compute("rot_fft")
            div_fft = self.compute("div_fft")
            urx_fft, ury_fft = self.oper.vecfft_from_rotfft(rot_fft)
            udx_fft, udy_fft = self.oper.vecfft_from_divfft(div_fft)
            ux_fft = urx_fft + udx_fft
            if mpi.rank == 0:
                ap_fft = self.state_spect.get_var("ap_fft")
                ux_fft[0, 0] = ap_fft[0, 0]
            result = ux_fft
            if SAVE_IN_DICT:
                key2 = "uy_fft"
                uy_fft = ury_fft + udy_fft
                if mpi.rank == 0:
                    am_fft = self.state_spect.get_var("am_fft")
                    uy_fft[0, 0] = am_fft[0, 0]

                self.vars_computed[key2] = uy_fft
                self.it_computed[key2] = it

        elif key == "uy_fft":
            rot_fft = self.compute("rot_fft")
            div_fft = self.compute("div_fft")
            urx_fft, ury_fft = self.oper.vecfft_from_rotfft(rot_fft)
            udx_fft, udy_fft = self.oper.vecfft_from_divfft(div_fft)
            uy_fft = ury_fft + udy_fft
            if mpi.rank == 0:
                am_fft = self.state_ff.get_var("am_fft")
                uy_fft[0, 0] = am_fft[0, 0]
            result = uy_fft
            if SAVE_IN_DICT:
                key2 = "ux_fft"
                ux_fft = urx_fft + udx_fft
                if mpi.rank == 0:
                    ap_fft = self.state_spect.get_var("ap_fft")
                    ux_fft[0, 0] = ap_fft[0, 0]
                self.vars_computed[key2] = ux_fft
                self.it_computed[key2] = it

        elif key == "rot":
            rot_fft = self.compute("rot_fft")
            result = self.oper.ifft2(rot_fft)

        elif key == "q_fft":
            result = self.oper.create_arrayK(value=0)

        elif key == "q":
            result = self.oper.create_arrayX(value=0)

        else:
            result = super().compute(
                key, SAVE_IN_DICT=SAVE_IN_DICT, RAISE_ERROR=RAISE_ERROR
            )
            SAVE_IN_DICT = False

        # to_print = 'Do not know how to compute "'+key+'".'
        # if RAISE_ERROR:
        #     raise ValueError(to_print)
        # else:
        #     if mpi.rank == 0:
        #         print(to_print
        #               +'\nreturn an array of zeros.')
        #     result = self.oper.create_arrayX(value=0.)

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

        return result

    def statespect_from_statephys(self):
        """Compute the state in Fourier space."""
        ux = self.state_phys.get_var("ux")
        uy = self.state_phys.get_var("uy")
        eta = self.state_phys.get_var("eta")

        eta_fft = self.oper.fft2(eta)
        ux_fft = self.oper.fft2(ux)
        uy_fft = self.oper.fft2(uy)

        (q_fft, ap_fft, am_fft) = self.oper.qapamfft_from_uxuyetafft(
            ux_fft, uy_fft, eta_fft
        )

        self.state_spect.set_var("ap_fft", ap_fft)
        self.state_spect.set_var("am_fft", am_fft)

    def statephys_from_statespect(self):
        """Compute the state in physical space."""
        ifft2 = self.oper.ifft2
        q_fft = self.oper.create_arrayK(value=0)
        ap_fft = self.state_spect.get_var("ap_fft")
        am_fft = self.state_spect.get_var("am_fft")

        (ux_fft, uy_fft, eta_fft) = self.oper.uxuyetafft_from_qapamfft(
            q_fft, ap_fft, am_fft
        )

        self.state_phys.set_var("ux", ifft2(ux_fft))
        self.state_phys.set_var("uy", ifft2(uy_fft))
        self.state_phys.set_var("eta", ifft2(eta_fft))

    def return_statephys_from_statespect(self, state_spect=None):
        """Return the state in physical space."""
        ifft2 = self.oper.ifft2
        if state_spect is None:
            state_spect = self.state_spect

        q_fft = self.oper.create_arrayK(value=0)
        ap_fft = state_spect.get_var("ap_fft")
        am_fft = state_spect.get_var("am_fft")

        (ux_fft, uy_fft, eta_fft) = self.oper.uxuyetafft_from_qapamfft(
            q_fft, ap_fft, am_fft
        )

        state_phys = SetOfVariables(like=self.state_phys)

        state_phys.set_var("ux", ifft2(ux_fft))
        state_phys.set_var("uy", ifft2(uy_fft))
        state_phys.set_var("eta", ifft2(eta_fft))

        return state_phys

    def init_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft):

        (q_fft, ap_fft, am_fft) = self.oper.qapamfft_from_uxuyetafft(
            ux_fft, uy_fft, eta_fft
        )

        state_spect = self.state_spect
        state_spect.set_var("ap_fft", ap_fft)
        state_spect.set_var("am_fft", am_fft)

        self.oper.dealiasing(state_spect)
        self.statephys_from_statespect()

    def init_from_etafft(self, eta_fft):
        (q_fft, ap_fft, am_fft) = self.oper.qapamfft_from_etafft(eta_fft)

        state_spect = self.state_spect
        state_spect.set_var("ap_fft", ap_fft)
        state_spect.set_var("am_fft", am_fft)

        self.oper.dealiasing(state_spect)
        self.statephys_from_statespect()

    def init_from_uxuyfft(self, ux_fft, uy_fft):

        oper = self.oper
        ifft2 = oper.ifft2

        oper.projection_perp(ux_fft, uy_fft)
        oper.dealiasing(ux_fft, uy_fft)

        ux = ifft2(ux_fft)
        uy = ifft2(uy_fft)

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        rot = ifft2(rot_fft)

        eta_fft = self._etafft_no_div(ux, uy, rot)
        eta = ifft2(eta_fft)

        (q_fft, ap_fft, am_fft) = self.oper.qapamfft_from_uxuyetafft(
            ux_fft, uy_fft, eta_fft
        )

        state_spect = self.state_spect
        state_spect.set_var("ap_fft", ap_fft)
        state_spect.set_var("am_fft", am_fft)

        state_phys = self.state_phys
        state_phys.set_var("ux", ux)
        state_phys.set_var("uy", uy)
        state_phys.set_var("eta", eta)
