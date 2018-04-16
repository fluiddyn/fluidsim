from __future__ import print_function

import unittest
import numpy as np

from fluidsim.solvers.ns2d.strat.output.test import TestNS2DStrat, mpi


class TestPrintStdsOut(TestNS2DStrat):
    _tag = "print_stdout"

    @unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
    def test_plot_print_stdout(self):
        self._plot()


if __name__ == "__main__":
    unittest.main()
