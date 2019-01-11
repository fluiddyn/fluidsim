import unittest
import shutil

from fluiddyn.io import stdout_redirected
from fluiddyn.util import mpi


class TestSimul(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.init_params()
        with stdout_redirected():
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
