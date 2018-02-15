"""Spherical harmonics operators (:mod:`fluidsim.operators.sphericalharmo`)
===========================================================================


"""

from fluiddyn.calcul.sphericalharmo import EasySHT

from fluidsim.base.params import Parameters


class OperatorsSphericalHarmonics(EasySHT):

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """

        attribs = {'lmax': 15,
                   'radius': 1.}
        params._set_child('oper', attribs=attribs)

    def __init__(self, params=None):

        if params is None:
            params = Parameters(tag='params')
            self.__class__._complete_params_with_default(params)


        lmax = params.oper.lmax
        radius = params.oper.radius

        super(OperatorsSphericalHarmonics, self).__init__(
            lmax=lmax, radius=radius)

        self._zeros_sh = self.create_array_sh(0.)

    def vec_from_rotsh(self, rot_sh):
        return self.uv_from_hdivrotsh(self._zeros_sh, rot_sh)

    def vec_from_divsh(self, div_sh):
        return self.uv_from_hdivrotsh(div_sh, self._zeros_sh)


if __name__ == '__main__':

    oper = OperatorsSphericalHarmonics()
