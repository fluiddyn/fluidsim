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

import numpy as np

from fluiddyn.io import stdout_redirected
from fluiddyn.util import mpi
from fluiddyn.util.compat import cached_property

try:
    import fluidfft
except ImportError:
    FLUIDFFT_INSTALLED = False
else:
    FLUIDFFT_INSTALLED = True


def skip_if_no_fluidfft(func):
    return unittest.skipUnless(FLUIDFFT_INSTALLED, "FluidFFT is not installed")(
        func
    )


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
                results = method()
        return results

    @unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
    def plot(self, name):
        """Test if plot methods work."""
        import matplotlib.pyplot as plt

        attr = self.get_sim_output_attr_from_str(name)
        attr.plot()
        plt.close("all")


class TimeLoggingTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_timings = []

    def startTest(self, test):
        self._test_started_at = time.time()
        super().startTest(test)

    def addSuccess(self, test):
        elapsed = time.time() - self._test_started_at
        name = self.getDescription(test)
        self.test_timings.append((name, elapsed))
        super().addSuccess(test)

    def getTestTimings(self):
        return self.test_timings


class TimeLoggingTestRunner(unittest.TextTestRunner):
    def __init__(self, slow_test_threshold=0.3, *args, **kwargs):
        self.slow_test_threshold = slow_test_threshold
        return super().__init__(
            resultclass=TimeLoggingTestResult, *args, **kwargs
        )

    def run(self, tests):
        result = super().run(tests)
        msg = f"\n\nSlow tests (>{self.slow_test_threshold:.3f}s):"
        self.write_result(msg)
        self.write_result("-" * len(msg))

        for name, elapsed in result.getTestTimings():
            if elapsed > self.slow_test_threshold:
                self.write_result(f"({elapsed:.3f}s) {name}")

        return result

    def write_result(self, *strs):
        """Write strings to the result stream."""
        if mpi.rank == 0:
            msg = " ".join(strs)
            self.stream.writeln(msg)


def _mname(obj):
    """Get the full dotted name of the test method"""

    mod_name = obj.__class__.__module__.replace("fluidsim.", "")
    return f"{mod_name}.{obj.__class__.__name__}.{obj._testMethodName}"


def deactivate_redirect_stdout(tests):
    TestCase.has_to_redirect_stdout = False


def _run(tests, verbose=False, no_capture=False):
    """Run a set of tests using unittest."""

    if mpi.rank == 0:
        verbosity = 2
    else:
        verbosity = 0

    if verbose:
        TestRunner = TimeLoggingTestRunner
    else:
        TestRunner = unittest.runner.TextTestRunner

    testRunner = TestRunner(verbosity=verbosity)

    if no_capture:
        deactivate_redirect_stdout(tests)

    result = testRunner.run(tests)
    if verbose:
        msg = "Skipped tests"
        testRunner.write_result("\n", msg, "\n", "-" * len(msg))
        for (case, reason) in result.skipped:
            testRunner.write_result("S  {} ({})".format(_mname(case), reason))
        for (case, reason) in result.expectedFailures:
            testRunner.write_result("X  %s" % _mname(case))

    return result


def import_test_module(module_name: str):
    """Smarter import handling common mistakes with warnings."""
    if not module_name.startswith("fluidsim"):
        warn(
            f"Assuming you forgot to add fluidsim in front of module name: {module_name}"
        )
        module_name = ".".join(("fluidsim", module_name))

    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        module_name = ".".join(module_name.split(".")[:-1])
        # warn(f"Module not found. Attempting {module_name} instead")
        module = import_module(module_name)
    finally:
        return module


def discover_tests(
    verbose=False, no_capture=False, module_name="fluidsim", path=None
):
    """Discovers all tests under a module or directory.
    Similar to `python -m unittest discover`.

    Parameters
    ----------
    module_name: str

        Tests are discovered under this module

    start_dir: str

        Tests are discovered under this directory. Overrides `module_name`
        parameter.

    verbose: bool

        For verbose output

    """
    if path is None:
        module = import_test_module(module_name)
        path_src = inspect.getfile(module)
        path = os.path.dirname(path_src)

    if os.path.isdir(path):
        loader = unittest.TestLoader()
        tests = loader.discover(path)
    else:
        module = path.replace(os.path.sep, ".")
        module = import_test_module(module)
        tests = unittest.defaultTestLoader.loadTestsFromModule(module)
        suite = unittest.TestSuite()
        suite.addTests(tests)

    return _run(tests, verbose, no_capture)


def collect_tests(verbose, no_capture, *modules):
    """Creates a `TestSuite` from several modules.

    Parameters
    ----------
    modules: str, str, ...

        Strings representing modules containing atleast one unittest.TestCase
        class.

    Examples
    --------
    >>> collect_tests(
            'fluidsim.solvers.test.test_ns',
            'fluidsim.solvers.test.test_sw1l',
            'fluidsim.operators.test.test_operators2d')

    """
    suite = unittest.TestSuite()
    for module in modules:
        module = module.replace(os.path.sep, ".")
        module = import_test_module(module)
        tests = unittest.defaultTestLoader.loadTestsFromModule(module)
        suite.addTests(tests)

    return _run(suite, verbose, no_capture)


def init_parser(parser):
    """Initialize argument parser for `fluidsim-test`."""

    parser.add_argument(
        "-m",
        "--module",
        default="fluidsim",
        help="tests are discovered under this module",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=None,
        help=(
            "tests are discovered under this directory/file and overrides -m"
            " option."
        ),
    )
    parser.add_argument(
        "-c",
        "--collect",
        nargs="+",
        default=[],
        help=(
            "create a TestSuite by collecting multiple modules containing"
            "TestCases"
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="for verbose output"
    )
    parser.add_argument(
        "-nc", "--no-capture", action="store_true", help="No stdout capture"
    )


def run(args=None):
    """Run fluidsim-test command."""

    if args is None:
        parser = argparse.ArgumentParser(
            prog="fluidsim-test",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Run FluidSim tests using unittest from any directory.",
        )

        init_parser(parser)
        args = parser.parse_args()

    if len(args.collect) > 0:
        result = collect_tests(args.verbose, args.no_capture, *args.collect)
    else:
        result = discover_tests(
            args.verbose, args.no_capture, args.module, args.path
        )

    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    run()
