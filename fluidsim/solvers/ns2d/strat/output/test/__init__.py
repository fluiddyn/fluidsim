from __future__ import print_function

import unittest
import numpy as np

from fluidsim.solvers.sw1l.output.test import BaseTestCase, mpi

class TestNS2DStrat(BaseTestCase):
    solver = 'NS2D.strat'
    _tag = ''

    @classmethod
    def setUpClass(cls, init_fields='noise',
                   type_forcing='tcrandom_anisotropic'):
        nh = 128
        super(TestNS2DStrat, cls).setUpClass(nh=nh, init_fields=init_fields,
                                             type_forcing=type_forcing,
                                             HAS_TO_SAVE=False,
                                             forcing_enable=False,
                                             dissipation_enable=False)
