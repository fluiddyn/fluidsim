"""Simple text output (:mod:`fluidsim.solvers.ns2d.strat.output.print_stdout`)
========================================================================

.. autoclass:: PrintStdOutNS2DStrat
   :members:
   :private-members:

"""

from __future__ import print_function, division

import numpy as np

from fluidsim.base.output.print_stdout import PrintStdOutBase

from fluiddyn.util import mpi

from fluidsim.solvers.ns2d.output.print_stdout import PrintStdOutNS2D

PrintStdOutNS2DStrat = PrintStdOutNS2D
