"""Coordinate systems in 3D (:mod:`fluidsim.operator.coord_system3d`)
=====================================================================

Provides:

.. autoclass:: CoordSystem3DConverter
   :members:
   :private-members:

"""

import numpy as np


class CoordSystem3DConverter:
    """Conversion from Cartesian coordinate system to cylindrical and spherical coordinate systems"""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        # cylindrical coordinates
        self.rh = np.sqrt(self.x**2 + self.y**2)
        # rtheta is defined between [-pi; pi] and is 0 for x = y = 0
        self.r_theta = np.atan2(self.y, self.x)

        # spherical coordinates
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def compute_cylindrical_components(self, vx, vy, vz):
        """
        Convert (vx, vy, vz) in cylindrical coordinates (vh, vt, vz)
        """

        with np.errstate(divide="ignore", invalid="ignore"):
            vh = np.where(self.rh != 0, (self.x * vx + self.y * vy) / self.rh, 0)
            vt = np.where(self.rh != 0, (-self.y * vx + self.x * vy) / self.rh, 0)

        return vh, vt, vz

    def compute_radial_component(self, vx, vy, vz):
        """
        Compute radial component vr along r
        """

        with np.errstate(divide="ignore", invalid="ignore"):
            vr = np.where(
                self.r != 0, (self.x * vx + self.y * vy + self.z * vz) / self.r, 0
            )

        return vr

    def compute_spherical_components(self, vx, vy, vz):
        """
        Convert vx, vy, vz in spherical coordinates (vr, vt, vp)
        """

        with np.errstate(divide="ignore", invalid="ignore"):
            self.rp = np.where(
                self.r != 0, np.arccos(np.clip(self.z / self.r, -1, 1)), 0
            )

            vr = np.where(
                self.r != 0, (self.x * vx + self.y * vy + self.z * vz) / self.r, 0
            )

            vt = np.where(self.rh != 0, (-self.y * vx + self.x * vy) / self.rh, 0)

            vp = np.where(
                self.r != 0,
                (self.z * (self.x * vx + self.y * vy) - self.rh**2 * vz)
                / (self.r * self.rh),
                0,
            )
            # Alternative for vp when rh = 0
            vp = np.where(self.rh != 0, vp, 0)

        return vr, vt, vp
