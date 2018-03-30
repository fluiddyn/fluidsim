"""Utilities for the unittests (:mod:`fluidsim.util.testing`)
=============================================================

This module extends the functionalities provide by the `unittest` standard
library. It enforces matplotlib to use its `Agg` backend, thereby smoothly
executing plotting tests without window pop-ups. It is also useful to execute
this module to run FluidSim unittests, without going into the source directory
for eg. when it is installed through `pip`.

"""
import os
import sys
import inspect
import unittest
import argparse
import time

import matplotlib
from importlib import import_module
from fluiddyn.util import mpi


matplotlib.use('agg')


class TimeLoggingTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super(TimeLoggingTestResult, self).__init__(*args, **kwargs)
        self.test_timings = []

    def startTest(self, test):
        self._test_started_at = time.time()
        super(TimeLoggingTestResult, self).startTest(test)

    def addSuccess(self, test):
        elapsed = time.time() - self._test_started_at
        name = self.getDescription(test)
        self.test_timings.append((name, elapsed))
        super(TimeLoggingTestResult, self).addSuccess(test)

    def getTestTimings(self):
        return self.test_timings


class TimeLoggingTestRunner(unittest.TextTestRunner):

    def __init__(self, slow_test_threshold=0.3, *args, **kwargs):
        self.slow_test_threshold = slow_test_threshold
        return super(TimeLoggingTestRunner, self).__init__(
            resultclass=TimeLoggingTestResult, *args, **kwargs)

    def run(self, test):
        result = super(TimeLoggingTestRunner, self).run(test)
        msg = "\n\nSlow tests (>{:.3f}s):".format(self.slow_test_threshold)
        self.write_result(msg)
        self.write_result('-' * len(msg))

        for name, elapsed in result.getTestTimings():
            if elapsed > self.slow_test_threshold:
                self.write_result("({:.3f}s) {}".format(elapsed, name))

        return result

    def write_result(self, *strs):
        """Write strings to the result stream."""
        if mpi.rank == 0:
            msg = ' '.join(strs)
            self.stream.writeln(msg)


def _mname(obj):
    """ Get the full dotted name of the test method """

    mod_name = obj.__class__.__module__.replace('fluidsim.', '')
    return "%s.%s.%s" % (mod_name, obj.__class__.__name__, obj._testMethodName)


def _run(tests, verbose=False):
    """Run a set of tests using unittest."""

    if verbose:
        testRunner = TimeLoggingTestRunner(verbosity=1)
    else:
        testRunner = unittest.runner.TextTestRunner(verbosity=1)

    result = testRunner.run(tests)
    if verbose:
        msg = 'Skipped tests'
        testRunner.write_result('\n', msg, '\n', '-' * len(msg))
        for (case, reason) in result.skipped:
            testRunner.write_result("S  %s (%s)" % (_mname(case), reason))
        for (case, reason) in result.expectedFailures:
            testRunner.write_result("X  %s" % _mname(case))

    return result


def discover_tests(module_name='fluidsim', start_dir=None, verbose=False):
    '''Discovers all tests under a module or directory.
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

    '''
    if start_dir is None:
        module = import_module(module_name)
        path_src = inspect.getfile(module)
        start_dir = os.path.dirname(path_src)

    loader = unittest.TestLoader()
    tests = loader.discover(start_dir)
    return _run(tests, verbose)


def collect_tests(verbose, *modules):
    '''Creates a `TestSuite` from several modules.

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

    '''
    suite = unittest.TestSuite()
    for module in modules:
        module = import_module(module)
        tests = unittest.defaultTestLoader.loadTestsFromModule(module)
        suite.addTests(tests)

    return _run(suite, verbose)


def init_parser(parser):
    """Initialize argument parser for `fluidsim-test`."""

    parser.add_argument(
        '-m', '--module', default='fluidsim',
        help='tests are discovered under this module')
    parser.add_argument(
        '-s', '--start-dir', default=None,
        help=(
            'tests are discovered under this directory and overrides -m'
            ' option.')
    )
    parser.add_argument(
        '-c', '--collect', nargs='+', default=[],
        help=('create a TestSuite by collecting multiple modules containing'
              'TestCases')
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='for verbose output')


def run(args=None):
    """Run fluidsim-test command."""

    if args is None:
        parser = argparse.ArgumentParser(
            prog='fluidsim-test',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Run FluidSim tests using unittest from any directory.')

        init_parser(parser)
        args = parser.parse_args()

    if len(args.collect) > 0:
        result = collect_tests(args.verbose, *args.collect)
    else:
        result = discover_tests(args.module, args.start_dir, args.verbose)

    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    run()
