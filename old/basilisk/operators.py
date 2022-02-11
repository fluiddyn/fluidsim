"""Operators for Basilisk (:mod:`fluidsim.base.basilisk.operators`)
===================================================================

Provides:

.. autoclass:: OperatorsBasilisk2D
   :members:
   :private-members:

"""

import numpy as np

import basilisk.stream as basilisk


class OperatorsBasilisk2D:
    """2D operators."""

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""

        attribs = {"nx": 48, "ny": 48, "Lx": 8, "Ly": 8}
        params._set_child("oper", attribs=attribs)

    def __init__(self, params=None, SEQUENTIAL=None):

        self.basilisk = basilisk

        nx = self.nx_seq = self.ny_seq = int(params.oper.nx)

        self.basilisk.init_grid(nx)

        self.params = params
        self.shapeX_seq = self.shapeX_loc = [nx, nx]

        self.x_seq = x = np.linspace(0, 1, nx)
        self.y_seq = y = np.linspace(0, 1, nx)
        self.X, self.Y = np.meshgrid(x, y)

    def produce_str_describing_oper(self):
        """Produce a string describing the operator."""

        return ""

    def produce_long_str_describing_oper(self):
        """Produce a string describing the operator."""

        return "2d Basilisk simulation\n"
