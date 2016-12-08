"""State for the NS2D.strat solver (:mod:`fluidsim.solvers.ns2d.strat.state`)
=================================================================

.. autoclass:: StateNS2DStrat
   :members:
   :private-members:

"""
from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi

from fluidsim.solvers.ns2d.state import StateNS2D

StateNS2DStrat = StateNS2D
