"""Utilities for the unittests (:mod:`fluidsim.util.testing`)
=============================================================

This module extends the functionalities provide by the `unittest` standard
library. It enforces matplotlib to use its `Agg` backend, thereby smoothly
executing plotting tests without window pop-ups. It is also useful to execute
this module to run FluidSim unittests, without going into the source directory
for eg. when it is installed through `pip`.

"""
import argparse
import inspect
import os
import shutil
import sys
import time
import unittest
from importlib import import_module
from warnings import warn
from pathlib import Path

import pytest

import numpy as np

from fluiddyn.io import stdout_redirected
from fluiddyn.util import mpi
from fluiddyn.util.compat import cached_property

import fluidsim

try:
    import fluidfft
except ImportError:
    FLUIDFFT_INSTALLED = False
else:
    FLUIDFFT_INSTALLED = True


def skip_if_no_fluidfft(func):
    return pytest.mark.skipif(
        not FLUIDFFT_INSTALLED, reason="FluidFFT is not installed"
    )(func)


class classproperty:
    """A combination of property and classmethod. Decorator that converts a
    method with a single cls argument into a property that can be accessed
    directly from the class.

    Borrowed from django.utils.functional

    License: BSD-3
    Copyright (c) Django Software Foundation and individual contributors.
    All rights reserved.

    """

    def __init__(self, method=None):
        self.fget = method

    def __get__(self, instance, cls=None):
        return self.fget(cls)

    def getter(self, method):
        self.fget = method
        return self


class TestCase(unittest.TestCase):

    # True except if pytest is used...
    has_to_redirect_stdout = not any(
        any(test_tool in arg for arg in sys.argv)
        for test_tool in ("pytest", "py.test")
    )

    def run(self, result=None):
        with stdout_redirected(self.has_to_redirect_stdout):
            super().run(result=result)


class TestSimul(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.init_params()
        with stdout_redirected(cls.has_to_redirect_stdout):
            cls.sim = cls.Simul(cls.params)

    @classmethod
    def tearDownClass(cls):
        # clean by removing the directory
        if mpi.rank == 0:
            if hasattr(cls, "sim"):
                shutil.rmtree(cls.sim.output.path_run, ignore_errors=True)

    @classmethod
    def init_params(cls):
        params = cls.params = cls.Simul.create_default_params()

        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"

        params.time_stepping.USE_CFL = False
        params.time_stepping.USE_T_END = False
        params.time_stepping.it_end = 2
        params.time_stepping.deltat0 = 0.1


class TestSimulConserve(TestSimul):
    """A test case which makes it easy to test for energy and enstrophy
    conservation. By default the simulation is instantiated and run till the end
    within the ``setUpClass`` method.

    """

    zero = 1e-14

    @classmethod
    def setUpClass(cls):
        cls.init_params()
        with stdout_redirected(cls.has_to_redirect_stdout), cls.Simul(
            cls.params
        ) as sim:
            cls.sim = sim
            sim.time_stepping.start()

    @cached_property
    def tendencies_fft(self):
        self.sim.params.forcing.enable = False
        return self.sim.tendencies_nonlin()

    def assertAlmostZero(
        self, value, places=None, msg=None, tolerance_warning=True
    ):
        """Assert if value is almost zero."""
        if places is None:
            places = -int(np.log10(self.zero))

        self.assertAlmostEqual(value, 0, places=places, msg=msg)
        if places < 7 and mpi.rank == 0 and tolerance_warning:
            warn(
                "Machine zero level too high. Value to be asserted as zero"
                f"= {value}\n\tTest: {self.id()}"
            )


class TestSimulConserveOutput(TestSimulConserve):
    """A test case with methods to easily test for the output modules."""

    def get_sim_output_attr_from_str(self, module):
        return getattr(self.sim.output, module)

    def get_results(self, name):
        module = self.get_sim_output_attr_from_str(name)
        for method_str in ("compute", "load_dataset", "load"):
            try:
                method = getattr(module, method_str)
            except AttributeError:
                pass
            else:
                try:
                    results = method()
                except NotImplementedError:
                    pass
        return results

    @unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
    def plot(self, name):
        """Test if plot methods work."""
        import matplotlib.pyplot as plt

        attr = self.get_sim_output_attr_from_str(name)
        attr.plot()
        plt.close("all")


def run():
    if len(sys.argv) == 1 or not Path(sys.argv[1]).exists():
        sys.argv.insert(1, str(Path(fluidsim.__file__).parent))

    if not any("--durations=" in arg for arg in sys.argv):
        sys.argv.append("--durations=10")

    return pytest.main()


if __name__ == "__main__":
    sys.exit(run())
