"""Operators finite differences (:mod:`fluidsim.operators.op_finitediff2d`)
===========================================================================

Provides:

.. autoclass:: OperatorFiniteDiff1DPeriodic
   :members:
   :private-members:

"""

import numpy as np

try:
    import scipy.sparse as sparse
except ImportError:
    pass

from .base import OperatorsBase1D


class OperatorFiniteDiff1DPeriodic(OperatorsBase1D):
    def __init__(self, params=None):
        super().__init__(params)
        nx = self.nx
        self.nx_seq = nx
        dx = self.deltax

        self.sparse_px = sparse.diags(
            diagonals=[-np.ones(nx - 1), np.ones(nx - 1), -1, 1],
            offsets=[-1, 1, nx - 1, -(nx - 1)],
        )
        self.sparse_px = self.sparse_px / (2 * dx)

        self.sparse_pxx = sparse.diags(
            diagonals=[np.ones(nx - 1), -2 * np.ones(nx), np.ones(nx - 1), 1, 1],
            offsets=[-1, 0, 1, nx - 1, -(nx - 1)],
        )

        self.sparse_pxx = self.sparse_pxx / dx**2

    def px(self, a):
        return self.sparse_px.dot(a.flat)

    def pxx(self, a):
        return self.sparse_pxx.dot(a.flat)

    def identity(self):
        return sparse.identity(self.size)

    def produce_long_str_describing_oper(self):
        return super().produce_long_str_describing_oper("Finite difference")
