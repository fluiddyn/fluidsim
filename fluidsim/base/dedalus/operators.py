"""Operators for Dedalus (:mod:`fluidsim.base.dedalus.operators`)
===================================================================

Provides:

.. autoclass:: OperatorsDedalus2D
   :members:
   :private-members:

"""

import numpy as np

from dedalus import public as dedalus


class OperatorsDedalus2D:
    """2D operators."""

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""

        attribs = {"nx": 128, "nz": 64, "Lx": 4, "Lz": 1}
        params._set_child("oper", attribs=attribs)

    def create_domain(self, params):
        de = dedalus
        par_oper = params.oper

        # Create bases and domain
        x_basis = de.Fourier(
            "x", par_oper.nx, interval=(0, par_oper.Lx), dealias=3 / 2
        )
        z_basis = de.Chebyshev(
            "z",
            par_oper.nz,
            interval=(-par_oper.Lz / 2, par_oper.Lz / 2),
            dealias=3 / 2,
        )
        return de.Domain([x_basis, z_basis], grid_dtype=np.float64)

    def __init__(self, params=None, SEQUENTIAL=None):

        self.domain = self.create_domain(params)

        self.axes = ("z", "x")
        self.Lx = params.oper.Lx
        self.Ly = params.oper.Lz

        self.params = params

        self.x_seq, self.y_seq = self.domain.grids(scales=1.0)
        self.x_seq = self.x_seq.flatten()
        self.y_seq = self.y_seq.flatten()
        self.X, self.Y = np.meshgrid(self.x_seq, self.y_seq)

        nx = self.nx_seq = len(self.x_seq)
        ny = self.ny_seq = len(self.y_seq)

        self.shapeX_seq = self.shapeX_loc = (ny, nx)

    def get_grid1d_seq(self, axe="x"):

        if axe not in ("x", "y"):
            raise ValueError

        return getattr(self, axe + "_seq")

    def produce_str_describing_oper(self):
        """Produce a string describing the operator."""

        return ""

    def produce_long_str_describing_oper(self):
        """Produce a string describing the operator."""

        return "2d Dedalus simulation\n"
