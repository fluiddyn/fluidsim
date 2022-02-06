import unittest
import os
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _show(*args, **kwargs):
    pass


plt.show = _show

os.environ["FLUIDSIM_TESTS_EXAMPLES"] = "1"

from fluiddyn.util import mpi
from fluidsim import FLUIDSIM_PATH

from runpy import run_path

def teardown_module(module):
    if mpi.rank == 0:
        shutil.rmtree(Path(FLUIDSIM_PATH) / "tests_examples", ignore_errors=True)

@unittest.skipIf(mpi.nb_proc > 1, "MPI not implemented")
def test_simul_ad1d():
    run_path("simul_ad1d.py")


def test_simul_ns2d():
    run_path("simul_ns2d.py")


def test_simul_ns2d_plot():
    run_path("simul_ns2d_plot.py")
