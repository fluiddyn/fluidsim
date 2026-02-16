"""Coordinates conversion in 3D (:mod:`fluidsim.operator.coord_convers3d`)
==============================================================

Provides:

.. autoclass:: OperConversCoord3D
   :members:
   :private-members:

"""

import numpy as np


class OperConversCoord3D:
    """Conversion from cartesian coordinates system to cylindrical, radial and spherical coordinate system"""

    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def compute_cylindrical_components(self, V):
        """
        Convert V = (Vx, Vy, Vz) in cylindrical coordiates (Vh, Vt, Vz)
        """
        Vx, Vy, Vz = V

        self.rh = np.sqrt(self.X**2 + self.Y**2)
        self.rt = np.atan2(
            self.Y, self.X
        )  # rtheta, defined between [-pi; pi] and is 0 for x = y = 0
        self.rv = self.Z

        with np.errstate(divide="ignore", invalid="ignore"):
            Vh = np.where(self.rh != 0, (self.X * Vx + self.Y * Vy) / self.rh, 0)
            Vt = np.where(self.rh != 0, (-self.Y * Vx + self.X * Vy) / self.rh, 0)

        return Vh, Vt, Vz

    def compute_radial_component(self, V):
        """
        Compute radial component Vr along r
        """
        Vx, Vy, Vz = V

        self.r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)

        with np.errstate(divide="ignore", invalid="ignore"):
            Vr = np.where(
                self.r != 0, (self.X * Vx + self.Y * Vy + self.Z * Vz) / self.r, 0
            )

        return Vr

    def compute_spherical_components(self, V):
        """
        Convert V = (Vx, Vy, Vz) in spherical coordinates (Vr, Vt, Vp)
        """
        Vx, Vy, Vz = V

        self.r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        self.rh = np.sqrt(self.X**2 + self.Y**2)  # projection dans le plan xy
        self.rt = np.atan2(self.Y, self.X)

        with np.errstate(divide="ignore", invalid="ignore"):

            self.rp = np.where(
                self.r != 0, np.arccos(np.clip(self.Z / self.r, -1, 1)), 0
            )

            Vr = np.where(
                self.r != 0, (self.X * Vx + self.Y * Vy + self.Z * Vz) / self.r, 0
            )

            Vt = np.where(self.rh != 0, (-self.Y * Vx + self.X * Vy) / self.rh, 0)

            Vp = np.where(
                self.r != 0,
                (self.Z * (self.X * Vx + self.Y * Vy) - self.rh**2 * Vz)
                / (self.r * self.rh),
                0,
            )
            # Alternative for Vp when rh = 0
            Vp = np.where(self.rh != 0, Vp, 0)

        return Vr, Vt, Vp