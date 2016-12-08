"""Spatial means output (:mod:`fluidsim.solvers.ns2d.strat.output.spatial_means`)
===========================================================================

.. autoclass:: SpatialMeansNS2DStrat
   :members:
   :private-members:

"""

from __future__ import division, print_function

import os
import numpy as np


from fluiddyn.util import mpi

from fluidsim.base.output.spatial_means import SpatialMeansBase

from fluidsim.solvers.ns2d.output.spatial_means import SpatialMeansNS2D

SpatialMeansNS2DStrat = SpatialMeansNS2D
