"""Energy budget (:mod:`fluidsim.solvers.ns2d.strat.output.spect_energy_budget`)
==========================================================================

.. autoclass:: SpectralEnergyBudgetNS2DStrat
   :members:
   :private-members:

"""

import numpy as np
import h5py


from fluidsim.base.output.spect_energy_budget import (
    SpectralEnergyBudgetBase, cumsum_inv)

from fluidsim.solvers.ns2d.output.spect_energy_budget import SpectralEnergyBudgetNS2D

SpectralEnergyBudgetNS2DStrat = SpectralEnergyBudgetNS2D
