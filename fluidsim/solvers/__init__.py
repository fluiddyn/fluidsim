"""Solvers for a variety of physical problems
=============================================

Base package containing the source code of the particular solvers.  Some
sub-packages contains also solvers for variant of the mentioned equations (for
example, :mod:`fluidsim.solvers.ns2d.strat` or
:mod:`fluidsim.solvers.sw1l.modified`).

.. autosummary::
   :toctree:

   ns2d
   ns3d
   sw1l
   plate2d
   ad1d
   waves2d
   sphere
   models0d
   nl1d
   burgers1d

Provides:

.. autosummary::
   :toctree:

   pkgload

"""
import numpy as _np
from warnings import warn as _warn


def pkgload():
    """Populate ``fluidsim.solvers`` package namespace with the solvers for easier
    import.

    """
    from ..util import available_solver_keys

    solvers = available_solver_keys()

    solver_pkgs = (f"fluidsim.solvers.{solver}" for solver in solvers)
    solver_modules = (f"fluidsim.solvers.{solver}.solver" for solver in solvers)

    _warn(
        (
            "This function will not work in numpy>=0.16.0 and will be removed in "
            "the future"
        ),
        FutureWarning,
    )
    _np.pkgload(*solver_pkgs)
    _np.pkgload(*solver_modules)
