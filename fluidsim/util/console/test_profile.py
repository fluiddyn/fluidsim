
import unittest
import sys
from glob import glob
from shutil import rmtree

import matplotlib
matplotlib.use('Agg')

from fluiddyn.util import mpi
from fluiddyn.io import stdout_redirected

from .__main__ import run_profile

path_dir = '/tmp/tmp_test_fluidsim_profile_dir'


class TestsProfile(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        if mpi.rank == 0:
            rmtree(path_dir)

    def test3d(self):
        with stdout_redirected():
            command = 'fluidsim-profile 8 -s ns3d -o'
            args = command.split()
            args.append(path_dir)
            sys.argv = args
            run_profile()
            paths = glob(path_dir + '/*')
            path = paths[0]

            if mpi.nb_proc == 1:
                command = 'fluidsim-profile -p -sf'
                args = command.split()
                args.append(path)
                sys.argv = args
                run_profile()

    def test2d(self):
        with stdout_redirected():
            command = 'fluidsim-profile 16 -s ns3d -o'
            args = command.split()
            args.append(path_dir)
            sys.argv = args
            run_profile()


if __name__ == '__main__':
    unittest.main()
