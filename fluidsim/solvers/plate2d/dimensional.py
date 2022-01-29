# -*- coding: utf-8 -*-

"""Plate2d solver (:mod:`fluidsim.solvers.plate2d.dimensional`)
================================================================

Provides:

.. autoclass:: Converter
   :members:
   :private-members:

"""

import numpy as np


class Converter:
    r"""Converter between dimensional and adimensional variables.

    Parameters
    ----------

    C : number
        Defined by the dispersion relation :math:`\omega = C k^2`.

    h : number
        Thickness of the plate (in meter).

    L : number, optional
        Length of the plate (in meter).

    sigma : number, optional
        Poisson modulus.

    Notes
    -----

    .. math::
       C = \frac{E h^2}{12\rho(1 - \sigma^2)}

    """

    def __init__(self, C, h, L=1.0, sigma=0.0):
        self.C = C
        self.h = h
        self.L = L
        self.sigma = sigma

        self.tilde_T = L**2 / np.sqrt(C)
        self.tilde_Z = h / np.sqrt(6 * (1 - sigma))
        self.tilde_L = L
        self.tilde_nu4 = L**4 / self.tilde_T

    def compute_time_adim(self, t):
        """Compute the dimensional time."""
        return t / self.tilde_T

    def compute_time_dim(self, t):
        """Compute the adimensional time."""
        return t * self.tilde_T

    def compute_z_adim(self, z):
        """Compute the dimensional deformation."""
        return z / self.tilde_Z

    def compute_z_dim(self, z):
        """Compute the adimensional deformation."""
        return z * self.tilde_Z

    def compute_w_adim(self, w):
        """Compute the dimensional deformation."""
        return self.tilde_T / self.tilde_Z * w

    def compute_w_dim(self, w):
        """Compute the adimensional deformation."""
        return self.tilde_Z / self.tilde_T * w

    def compute_nu4_adim(self, nu4):
        """Compute the dimensional hyper-viscosity."""
        return nu4 / self.tilde_nu4

    def compute_nu4_dim(self, nu4):
        """Compute the adimensional hyper-viscosity."""
        return nu4 * self.tilde_nu4


if __name__ == "__main__":

    converter = Converter(C=0.648**2, h=4e-4)

    amplitude_z = 0.001
    print("amplitude z: {:6.2f}".format(converter.compute_z_adim(amplitude_z)))

    nu_4 = 5e-7
    print("nu_4: {:8.3g}".format(converter.compute_nu4_adim(nu_4)))
