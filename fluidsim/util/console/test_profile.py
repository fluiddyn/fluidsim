import unittest
import sys
from glob import glob
from shutil import rmtree
import os

from fluiddyn.util import mpi

from fluidsim.util.testing import TestCase

from .__main__ import run_profile


if mpi.rank == 0:
    pid = os.getpid()
else:
    pid = None

if mpi.nb_proc > 1:
    pid = mpi.comm.bcast(pid, root=0)

path_dir = f"/tmp/tmp_test_fluidsim_profile_dir_pid{pid}"


class TestsProfile(TestCase):
    @classmethod
    def tearDownClass(cls):
        if mpi.rank == 0:
            rmtree(path_dir)

    def test3d(self):
        command = "fluidsim-profile 8 -s ns3d -o"
        args = command.split()
        args.append(path_dir)
        sys.argv = args
        run_profile()

        if mpi.nb_proc == 1:
            command = "fluidsim-profile -p -sf"
            args = command.split()
            paths = glob(path_dir + "/*")
            path = paths[0]
            args.append(path)
            sys.argv = args
            run_profile()

    def test2d(self):
        command = "fluidsim-profile 16 -s ns3d -o"
        args = command.split()
        args.append(path_dir)
        sys.argv = args
        run_profile()


if __name__ == "__main__":
    unittest.main()
