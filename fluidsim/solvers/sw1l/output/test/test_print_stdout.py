from __future__ import print_function

import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from fluidsim.solvers.sw1l.output.test import BaseTestCase, mpi


class TestPrintStdout(BaseTestCase):
    _tag = "print_stdout"

    @unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
    def test_plot_print_stdout(self):
        self._plot()

    def test_energy_vs_spatial_means(self):
        """Verify energy saved by spatial_means module is the same."""
        if mpi.nb_proc > 1:
            mpi.comm.barrier()
        df = self.output.spatial_means.load()

        # ignore last row to be comparable to print_stdout
        imax = -1 if len(df) > 1 else None
        assert_array_almost_equal(
            self.dict_results["E"], df.E[:imax].values, decimal=4
        )


if __name__ == "__main__":
    unittest.main()
