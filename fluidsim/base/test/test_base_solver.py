

import unittest

import numpy as np

import fluiddyn as fld

# to get fld.show
import fluiddyn.output

from fluidsim.base.solvers.pseudo_spect import SimulBasePseudoSpectral

from fluiddyn.io import stdout_redirected


class TestBaseSolver(unittest.TestCase):
    def test_simul(self):
        """Should be able to run a base experiment."""

    params = SimulBasePseudoSpectral.create_default_params()

    params.short_name_type_run = 'test'

    nh = 16
    Lh = 2*np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    params.nu_2 = 1.

    params.time_stepping.t_end = 2.

    params.output.periods_plot.phys_fields = 0.

    with stdout_redirected():
        sim = SimulBasePseudoSpectral(params)
        sim.time_stepping.start()

    fld.show()

if __name__ == '__main__':
    unittest.main()
