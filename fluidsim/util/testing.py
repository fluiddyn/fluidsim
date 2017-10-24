"""Utilities for the unittests (:mod:`fluidsim.util.testing`)
=============================================================
This module extends the functionalities provide by the `unittest` standard
library. It enforces matplotlib to use its `Agg` backend, thereby smoothly
executing plotting tests without window pop-ups. It is also useful to execute
this module to run FluidSim unittests, without going into the source directory
for eg. when it is installed through `pip`.

.. TODO: Use `argparse` to add optional arguments such as module names and
verbosity.

"""
import os
import sys
import inspect
import unittest

import matplotlib
from importlib import import_module
from fluiddyn.util import mpi


matplotlib.use('agg')


def _mname(obj):
    """ Get the full dotted name of the test method """

    mod_name = obj.__class__.__module__.replace('fluidsim.', '')
    return "%s.%s.%s" % (mod_name, obj.__class__.__name__, obj._testMethodName)


def _run(tests, verbose=False):
    """Run a set of tests using unittest."""

    testRunner = unittest.runner.TextTestRunner(verbosity=1)
    result = testRunner.run(tests)
    if verbose:
        for (case, reason) in result.skipped:
            mpi.printby0("S  %s (%s)" % (_mname(case), reason), file=sys.stderr)
        for (case, reason) in result.expectedFailures:
            mpi.printby0("X  %s" % _mname(case), file=sys.stderr)

    return result


def discover_tests(module_name='fluidsim', start_dir=None, verbose=False):
    '''Discovers all tests under a module or directory.
    Similar to `python -m unittest discover`.

    Parameters
    ----------
    module_name: str

        Tests are discovered under module

    start_dir: str

        Tests are discovered under this directory. Overrides `module_name`
        parameter.

    verbose: bool

        For verbose output

    '''
    if start_dir is None:
        module = import_module(module_name)
        path_src = inspect.getfile(module)
        start_dir = os.path.dirname(path_src)

    loader = unittest.TestLoader()
    tests = loader.discover(start_dir)
    return _run(tests, verbose)


def collect_tests(*modules):
    '''Creates a `TestSuite` from several modules.

    Parameters
    ----------
    modules: str, str, ...

        Strings representing modules containing atleast one unittest.TestCase
        class.

    Example
    -------
    >>> collect_tests(
            'fluidsim.solvers.test.test_solvers',
            'fluidsim.operators.test.test_operators2d')

    '''
    suite = unittest.TestSuite()
    for module in modules:
        module = import_module(module)
        tests = unittest.defaultTestLoader.loadTestsFromModule(module)
        suite.addTests(tests)

    return _run(suite)


if __name__ == '__main__':
    result = discover_tests()
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)
