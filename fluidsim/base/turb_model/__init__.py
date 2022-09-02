"""Turbulent models (:mod:`fluidsim.base.turb_model`)
=====================================================

Provides:

.. autosummary::
   :toctree:

   base
   smagorinsky
   stress_tensor

Examples
--------

.. code-block:: python

    from fluidsim.solvers.ns3d.solver import Simul
    from fluidsim.base.turb_model import extend_simul_class, SmagorinskyModel

    Simul = extend_simul_class(Simul, SmagorinskyModel)
    params = Simul.create_default_params()

    params.turb_model.enable = True
    params.turb_model.type = "smagorinsky"
    params.turb_model.smagorinsky.C = 0.18

"""

from fluidsim.extend_simul import extend_simul_class

from .smagorinsky import SmagorinskyModel


__all__ = ["extend_simul_class", "SmagorinskyModel"]
