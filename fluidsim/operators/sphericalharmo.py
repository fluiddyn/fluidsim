"""Spherical harmonics operators (:mod:`fluidsim.operators.sphericalharmo`)
===========================================================================


"""
import numpy as np

from pyshtools.constants import Earth

from fluiddyn.util.compat import cached_property

from fluidsht.sht2d.operators import OperatorsSphereHarmo2D as _Operator

# from fluiddyn.calcul.sphericalharmo import EasySHT as _Operator

from fluidsim.base.params import Parameters


class OperatorsSphericalHarmonics(_Operator):
    @staticmethod
    def _complete_params_with_default(params):
        f"""This static method is used to complete the *params* container.
        Default parameters

        - omega (solid body rotation speed)
        - radius (of the sphere)

        are initialized as per *{Earth.wgs84.r3.reference}*

        """

        attribs = {
            "lmax": 15,
            "nlat": None,
            "nlon": None,
            "omega": Earth.wgs84.omega.value,
            "radius": Earth.wgs84.r3.value,
        }
        params._set_child("oper", attribs=attribs)

    def __init__(self, params=None):
        self.axes = ("lat", "lon")
        if params is None:
            params = Parameters(tag="params")
            self.__class__._complete_params_with_default(params)

        self.params = params = params.oper
        super().__init__(
            lmax=params.lmax,
            nlat=params.nlat,
            nlon=params.nlon,
            radius=params.radius,
        )

    @cached_property
    def f_radial(self):
        r"""The vertical / radial component of the solid body rotation vector
        pointed along axis of rotation. Useful for calculation of coriolis
        term where :math:`f = 2 \Omega \sin \theta`.

        """
        return 2.0 * self.params.omega * np.sin(self.LATS)

    def get_grid1d_seq(self, axe="lat"):
        if axe not in self.axes:
            raise ValueError

        # TODO: implement
        # if self.params.ONLY_COARSE_OPER:
        # set_grid based on params.nlat and params.nlon and other flags
        # rebuild lats and lons
        # else:
        return getattr(self, axe + "s")


if __name__ == "__main__":

    oper = OperatorsSphericalHarmonics()
