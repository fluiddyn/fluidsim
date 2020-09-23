import os
import sys
import unittest
from glob import glob
from shutil import rmtree

from fluiddyn.util import mpi
from fluidsim.util.testing import TestCase, skip_if_no_fluidfft

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
        if mpi.rank == 0 and os.path.isdir(path_dir):
            rmtree(path_dir)

    @skip_if_no_fluidfft
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

    @skip_if_no_fluidfft
    def test2d(self):
        command = "fluidsim-profile 16 -s ns2d -o"
        args = command.split()
        args.append(path_dir)
        sys.argv = args
        run_profile()

    @unittest.skipIf(
        mpi.nb_proc > 1, "Profiling solver ad1d is not meant to be done with MPI"
    )
    def test1d(self):
        command = "fluidsim-profile 128 -d1 -s ad1d -o"
        args = command.split()
        args.append(path_dir)
        sys.argv = args

        # FIXME: works with pytest, but not with unittest
        #  from fluidsim.util.console.util import ConsoleError
        #  with self.assertRaises(ConsoleError):
        #      # No profiling implemented for 1D solvers
        #      run_profile()


if __name__ == "__main__":
    unittest.main()
