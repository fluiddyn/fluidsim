

import unittest

import fluiddyn as fld

# to get fld.show
import fluiddyn.output

from fluidsim.base.solvers.base import SimulBase as Simul

from fluiddyn.io import stdout_redirected


class TestBaseSolver(unittest.TestCase):
    def test_simul(self):
        """Should be able to run a base experiment."""

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'
    params.time_stepping.USE_CFL = False
    params.time_stepping.t_end = 2.
    params.time_stepping.deltat0 = 0.1

    with stdout_redirected():
        sim = Simul(params)
        sim.time_stepping.start()

    fld.show()


if __name__ == '__main__':
    unittest.main()
