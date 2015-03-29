import unittest
import shutil

import numpy as np

import fluiddyn as fld

from fluiddyn.io import stdout_redirected


class TestSolverNS2D(unittest.TestCase):
    def test_tendency(self):

        key_solver = 'NS2D'
        solver = fld.simul.import_module_solver_from_key(key_solver)
        params = fld.simul.create_params(solver)

        params.short_name_type_run = 'test'

        nh = 64
        params.oper.nx = nh
        params.oper.ny = nh
        Lh = 6.
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.oper.coef_dealiasing = 2./3
        params.nu_8 = 2.

        params.oper.type_fft = 'FFTWPY'
        # params.oper.type_fft = 'FFTWCY'

        params.time_stepping.t_end = 0.5

        params.init_fields.type_flow_init = 'NOISE'
        params.output.HAS_TO_SAVE = False
        params.FORCING = False

        with stdout_redirected():
            sim = solver.Simul(params)

        rot_fft = sim.state('rot_fft')

        tend = sim.tendencies_nonlin(state_fft=sim.state.state_fft)
        Frot_fft = tend['rot_fft']

        T_rot = np.real(Frot_fft.conj()*rot_fft)

        ratio = (sim.oper.sum_wavenumbers(T_rot) /
                 sim.oper.sum_wavenumbers(abs(T_rot)))

        self.assertGreater(1e-16, ratio)

        # print ('sum(T_rot) = {0:9.4e} ; '
        #        'sum(abs(T_rot)) = {1:9.4e}').format(
        #            sim.oper.sum_wavenumbers(T_rot),
        #            sim.oper.sum_wavenumbers(abs(T_rot)))

        # clean by removing the directory
        shutil.rmtree(sim.output.path_run)


if __name__ == '__main__':
    unittest.main()
