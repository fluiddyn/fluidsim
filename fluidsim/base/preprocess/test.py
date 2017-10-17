import unittest
from shutil import rmtree
import numpy as np

from fluidsim.base.solvers.pseudo_spect import SimulBasePseudoSpectral
import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected


class TestPreprocessPS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Should be able to run a base experiment."""
        cls.params = params = SimulBasePseudoSpectral.create_default_params()
        params.short_name_type_run = 'test_preprocess_ps'
        nh = 16
        Lh = 2 * np.pi
        params.oper.nx = nh
        params.oper.ny = nh
        params.oper.Lx = Lh
        params.oper.Ly = Lh
        params.init_fields.type = 'constant'
        params.preprocess.enable = True
        params.preprocess.viscosity_type = 'laplacian'
        params.preprocess.viscosity_scale = 'energy'

        with stdout_redirected(), SimulBasePseudoSpectral(params) as sim:
            cls.sim = sim

    @classmethod
    def tearDownClass(cls):
        if mpi.rank == 0:
            rmtree(cls.sim.output.path_run)

    def test_set_viscosity(self):
        """Tests if preprocess can initialize viscosity or not."""
        params = self.params
        sim = self.sim
        self.assertGreater(sim.params.nu_2, 0)
        self.assertEqual(sim.params.nu_4 + sim.params.nu_8, 0)
        assert np.any(sim.time_stepping.freq_lin > 0)


if __name__ == '__main__':
    unittest.main()
