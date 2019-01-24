"""State class for the sw1l.modified solver
(:mod:`fluidsim.solvers.sw1l.modified.state`)
===================================================


Provides:

.. autoclass:: StateSW1LModified
   :members:
   :private-members:

"""

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.solvers.sw1l.state import StateSW1L

from fluiddyn.util import mpi


class StateSW1LModified(StateSW1L):
    """Contains the variables corresponding to the state and handles the
    access to other fields for the solver MSW1L.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": ["ux_fft", "uy_fft", "eta_fft"],
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

        if key == "ux_fft":
            result = self.oper.fft2(self.state_phys.get_var("ux"))
        elif key == "uy_fft":
            result = self.oper.fft2(self.state_phys.get_var("ux"))
        elif key == "rot_fft":
            ux_fft = self.compute("ux_fft")
            uy_fft = self.compute("uy_fft")
            result = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        elif key == "div_fft":
            ux_fft = self.compute("ux_fft")
            uy_fft = self.compute("uy_fft")
            result = self.oper.divfft_from_vecfft(ux_fft, uy_fft)
        elif key == "rot":
            rot_fft = self.compute("rot_fft")
            result = self.oper.ifft2(rot_fft)
        elif key == "div":
            div_fft = self.compute("div_fft")
            result = self.oper.ifft2(div_fft)
        elif key == "q":
            rot = self.compute("rot")
            eta = self.sim.state.state_phys.get_var("eta")
            result = rot - self.params.f * eta
        elif key == "q_fft":
            ux_fft = self.state_spect.get_var("ux_fft")
            uy_fft = self.state_spect.get_var("uy_fft")
            eta_fft = self.state_spect.get_var("eta_fft")
            rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
            result = rot_fft - self.params.f * eta_fft
        elif key == "a_fft":
            ux_fft = self.state_spect.get_var("ux_fft")
            uy_fft = self.state_spect.get_var("uy_fft")
            eta_fft = self.state_spect.get_var("eta_fft")
            result = self.oper.afft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)
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

        state_spect = self.state_spect
        state_spect.set_var("ux_fft", ux_fft)
        state_spect.set_var("uy_fft", uy_fft)
        state_spect.set_var("eta_fft", eta_fft)

        state_phys = self.state_phys
        state_phys.set_var("ux", ux)
        state_phys.set_var("uy", uy)
        state_phys.set_var("eta", eta)

    def statephys_from_statespect(self):
        """Compute the state in physical space."""
        ifft2 = self.oper.ifft2
        ux_fft = self.state_spect.get_var("ux_fft")
        uy_fft = self.state_spect.get_var("uy_fft")
        eta_fft = self.state_spect.get_var("eta_fft")
        self.state_phys.set_var("ux", ifft2(ux_fft))
        self.state_phys.set_var("uy", ifft2(uy_fft))
        self.state_phys.set_var("eta", ifft2(eta_fft))

    def return_statephys_from_statespect(self, state_spect=None):
        """Return the state in physical space."""
        ifft2 = self.oper.ifft2
        if state_spect is None:
            state_spect = self.state_spect
        ux_fft = state_spect.get_var("ux_fft")
        uy_fft = state_spect.get_var("uy_fft")
        eta_fft = state_spect.get_var("eta_fft")
        state_phys = SetOfVariables(like=self.state_phys)
        state_phys.set_var("ux", ifft2(ux_fft))
        state_phys.set_var("uy", ifft2(uy_fft))
        state_phys.set_var("eta", ifft2(eta_fft))

        return state_phys
