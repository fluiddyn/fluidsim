from __future__ import print_function

import unittest
import numpy as np

from fluidsim.solvers.sw1l.output.test import BaseTestCase, mpi


class TestNS2DStrat(BaseTestCase):
    solver = "NS2D.strat"
    _tag = ""
    options = {
        "nh": 32,
        "init_fields": "noise",
        "type_forcing": "tcrandom_anisotropic",
        "HAS_TO_SAVE": True,
        "forcing_enable": False,
        "dissipation_enable": False,
        "periods_save_spatial_means": 0.25,
    }

    @classmethod
    def setUpClass(cls):
        super(TestNS2DStrat, cls).setUpClass(**cls.options)
