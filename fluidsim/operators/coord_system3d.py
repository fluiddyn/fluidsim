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

        # tiny constant
        EPSILON = 1e-12

        # cylindrical coordinates
        self.rh = np.sqrt(self.x**2 + self.y**2)
        self.rh_not0 = np.where(self.rh != 0, self.rh, EPSILON)

        # spherical coordinates
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.r_not0 = np.where(r != 0, r, EPSILON)

    def compute_r_theta(self):
        """r_theta is defined between [-pi; pi] and is 0 for x = y = 0"""
        return np.atan2(self.y, self.x)

    def compute_cylindrical_components(self, vx, vy, vz):
        """Convert (vx, vy, vz) in cylindrical coordinates (vh, vt, vz)"""
        vh = (self.x * vx + self.y * vy) / self.rh_not0
        vt = (-self.y * vx + self.x * vy) / self.rh_not0
        return vh, vt, vz

    def compute_radial_component(self, vx, vy, vz):
        """Compute (spherical) radial component"""
        return (self.x * vx + self.y * vy + self.z * vz) / self.r_not0

    def compute_spherical_components(self, vx, vy, vz):
        """Convert (vx, vy, vz) in spherical coordinates (vr, vt, vp)"""
        vr = self.compute_radial_component(vx, vy, vz)
        vt = (-self.y * vx + self.x * vy) / self.rh_not0
        vp = (self.z * (self.x * vx + self.y * vy) - self.rh**2 * vz) / (
            self.r_not0 * self.rh_not0
        )
        return vr, vt, vp
