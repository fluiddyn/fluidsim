"""Plate2d state (:mod:`fluidsim.solvers.plate2d.state`)
==============================================================
"""

import numpy as np

from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi


class StatePlate2D(StatePseudoSpectral):
    """Contains the variables corresponding to the state and handles the
    access to other fields for the solver PLATE2D.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": ["w_fft", "z_fft"],
                "keys_state_phys": ["w", "z"],
                "keys_computable": ["chi_fft", "chi", "Nw_fft", "lapz_fft"],
                "keys_phys_needed": ["w", "z"],
                "keys_linear_eigenmodes": [],
            }
        )

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        oper = self.oper
        it = self.sim.time_stepping.it
        if key in self.vars_computed and it == self.it_computed[key]:
            return self.vars_computed[key]

        if key == "chi_fft":
            z_fft = self.state_spect.get_var("z_fft")
            mamp_zz = oper.monge_ampere_from_fft(z_fft, z_fft)
            result = oper.invlaplacian_fft(
                oper.fft2(mamp_zz), order=4, negative=True
            )
        elif key == "chi":
            chi_fft = self.compute("chi_fft")
            result = oper.ifft2(chi_fft)
        elif key == "Nw_fft":
            mamp_zchi = oper.monge_ampere_from_fft(
                self.state_spect.get_var("z_fft"), self.compute("chi_fft")
            )
            result = oper.fft2(mamp_zchi)
        elif key == "lapz_fft":
            z_fft = self.state_spect.get_var("z_fft")
            result = oper.laplacian_fft(z_fft, order=4)
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
        w_fft = self.state_spect.get_var("w_fft")
        z_fft = self.state_spect.get_var("z_fft")

        w = self.state_phys.get_var("w")
        z = self.state_phys.get_var("z")

        self.oper.ifft_as_arg(w_fft, w)
        self.oper.ifft_as_arg(z_fft, z)

    def init_state_from_wz_fft(self, w_fft, z_fft):
        self.oper.dealiasing(w_fft, z_fft)
        self.state_spect.set_var("w_fft", w_fft)
        self.state_spect.set_var("z_fft", z_fft)
        self.statephys_from_statespect()

    def init_statespect_from(self, **kwargs):
        if len(kwargs) == 1:
            if "w_fft" in kwargs:
                w_fft = kwargs["w_fft"]
                z_fft = np.zeros_like(w_fft)
                self.init_state_from_wz_fft(w_fft, z_fft)
        else:
            super().init_statespect_from(**kwargs)
