"""Plate2d state (:mod:`fluidsim.solvers.plate2d.state`)
==============================================================
"""

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
        info_solver.classes.State._set_attribs({
            'keys_state_fft': ['w_fft', 'z_fft'],
            'keys_state_phys': ['w', 'z'],
            'keys_computable': [],
            'keys_phys_needed': ['w', 'z'],
            'keys_linear_eigenmodes': []})

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        oper = self.oper
        it = self.sim.time_stepping.it
        if (key in self.vars_computed and
                it == self.it_computed[key]):
            return self.vars_computed[key]

        if key == 'w_fft':
            result = oper.fft2(self.state_phys.get_var('w'))
        elif key == 'z_fft':
            result = oper.fft2(self.state_phys.get_var('z'))
        elif key == 'chi_fft':
            z_fft = self.state_fft.get_var('z_fft')
            mamp_zz = oper.monge_ampere_from_fft(z_fft, z_fft)
            result = - oper.invlaplacian2_fft(oper.fft2(mamp_zz))
        elif key == 'chi':
            chi_fft = self.compute('chi_fft')
            result = oper.ifft2(chi_fft)
        elif key == 'Nw_fft':
            mamp_zchi = oper.monge_ampere_from_fft(
                self.state_fft.get_var('z_fft'), self.compute('chi_fft'))
            result = oper.fft2(mamp_zchi)
        elif key == 'lapz_fft':
            z_fft = self.compute('z_fft')
            result = oper.laplacian2_fft(z_fft)
        else:
            to_print = 'Do not know how to compute "'+key+'".'
            if RAISE_ERROR:
                raise ValueError(to_print)
            else:
                if mpi.rank == 0:
                    print(to_print + '\nreturn an array of zeros.')

                result = self.oper.constant_arrayX(value=0.)

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

        return result

    def statephys_from_statefft(self):
        w_fft = self.state_fft.get_var('w_fft')
        z_fft = self.state_fft.get_var('z_fft')
        self.state_phys.set_var('w', self.oper.ifft2(w_fft))
        self.state_phys.set_var('z', self.oper.ifft2(z_fft))

    def init_state_from_wz_fft(self, w_fft, z_fft):
        self.sim.oper.dealiasing(w_fft, z_fft)
        self.state_fft.set_var('w_fft', w_fft)
        self.state_fft.set_var('z_fft', z_fft)
        self.statephys_from_statefft()
