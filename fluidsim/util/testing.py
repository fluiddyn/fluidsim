import inspect
import os
import unittest

import matplotlib
import fluidsim


matplotlib.use('agg')


def discover(start_dir=None):
    if start_dir is None:
        path_src = inspect.getfile(fluidsim)
        start_dir = os.path.dirname(path_src)

    loader = unittest.TestLoader()
    tests = loader.discover(start_dir)
    testRunner = unittest.runner.TextTestRunner()
    testRunner.run(tests)


if __name__ == '__main__':
    discover()
