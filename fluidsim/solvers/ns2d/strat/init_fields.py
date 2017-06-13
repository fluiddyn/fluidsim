"""Initialization of the field (:mod:`fluidsim.solvers.ns2d.strat.init_fields`)
===============================================================================

.. autoclass:: InitFieldsNS2D
   :members:

.. autoclass:: InitFieldsNoise
   :members:

.. autoclass:: InitFieldsJet
   :members:

.. autoclass:: InitFieldsDipole
   :members:

"""

import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields

from fluidsim.solvers.ns2d.init_fields import InitFieldsNS2D, InitFieldsNoise, InitFieldsJet, InitFieldsDipole

InitFieldsNS2DStrat = InitFieldsNS2D

InitFieldsNoiseStrat = InitFieldsNoise

InitFieldsJetStrat = InitFieldsJet

InitFieldsDipoleStrat = InitFieldsDipole
