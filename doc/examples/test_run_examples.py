import unittest
import os
from pathlib import Path
import shutil

import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _show(*args, **kwargs):
    pass


plt.show = _show

os.environ["FLUIDSIM_TESTS_EXAMPLES"] = "1"
os.environ["FLUIDSIM_PATH"] = "/tmp/tests_fluidsim"

from fluiddyn.util import mpi
from fluidsim import path_dir_results

from runpy import run_path


def teardown_module(module):
    if mpi.rank == 0:
        shutil.rmtree(path_dir_results / "examples", ignore_errors=True)
        shutil.rmtree(path_dir_results / "examples_restart", ignore_errors=True)


@unittest.skipIf(mpi.nb_proc > 1, "MPI not implemented")
def test_simul_ad1d():
    run_path("simul_ad1d.py")


@pytest.mark.parametrize(
    "script",
    [
        "simul_ns2dstrat_initfields_in_script.py",
        "simul_ns2d_plot.py",
        "simul_ns2d_plotphys.py",
        "simul_ns2dstrat_forcing_in_script.py",
        "simul_ns2dbouss_staticlayer.py",
        "simul_ns2dstrat_forcing_const_energy_rate.py",
        "simul_ns2dstrat_forcing_random.py",
        "simul_ns2dstrat_quasi1d.py",
        "simul_ns3dbouss_initfields_in_script.py",
        "simul_ns2dbouss_initfields_in_script.py",
        "simul_ns3dbouss_staticlayer.py",
        "simul_ns3dstrat_check_energy_conserv.py",
        "simul_ns3dstrat_forcing_random.py",
        "simul_ns3dstrat_waves.py",
    ],
)
def test_script(script):
    run_path(script)


def test_restart():
    run_path("simul_ns2d.py")
    run_path("simul_ns2d_restart.py")
