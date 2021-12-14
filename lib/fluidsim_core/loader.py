"""Utilities for loading and importing solvers

.. autofunction:: available_solvers
.. autofunction:: import_module_solver
.. autofunction:: import_cls_simul

"""

import entrypoints


def available_solvers(entrypoint_grp):
    """Returns a dictionary of all registered solver modules registered as an
    entrypoint_ group. Each entrypoint_ would be a key-value pair - a solver
    short name and the full import path of the solver module respectively.

    .. _entrypoint: https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata

    Parameters
    ----------
    entrypoint_grp: str
        The name of the entrypoint group listing the solvers.

    """  # noqa
    return entrypoints.get_group_named(entrypoint_grp)


def import_module_solver(key, entrypoint_grp):
    """Import the solver module.

    Parameters
    ----------
    key: str
        The short name of a solver.

    entrypoint_grp: str
        The name of the entrypoint group listing the solvers.

    """
    solvers = available_solvers(entrypoint_grp)
    try:
        solver = solvers[key].load()
    except KeyError:
        raise ValueError(
            "You have to give a proper solver key. Given: "
            f"{key}. Expected one of: {list(solvers)}"
        )
    else:
        return solver


def import_cls_simul(key, entrypoint_grp):
    """Import the Simul class of a solver.

    Parameters
    ----------
    key: str
        The short name of a solver.

    entrypoint_grp: str
        The name of the entrypoint group listing the solvers.

    """

    if key.startswith(entrypoint_grp + "."):
        key = key[len(entrypoint_grp) + 1 :]

    solver = import_module_solver(key, entrypoint_grp)
    return solver.Simul
