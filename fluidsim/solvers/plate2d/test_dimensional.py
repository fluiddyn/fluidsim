from .dimensional import Converter

from fluidsim.util.testing import TestCase


class TestDimensional(TestCase):
    def test_converter(self):

        conv = Converter(C=0.648**2, h=4e-4)

        time = 10.0
        time_adim = conv.compute_time_adim(time)
        self.assertEqual(time, conv.compute_time_dim(time_adim))

        amplitude_z = 0.001
        z_adim = conv.compute_z_adim(amplitude_z)
        self.assertEqual(amplitude_z, conv.compute_z_dim(z_adim))

        amplitude_w = 0.001
        w_adim = conv.compute_w_adim(amplitude_w)
        self.assertEqual(amplitude_w, conv.compute_w_dim(w_adim))

        nu4 = 5e-7
        nu4_adim = conv.compute_nu4_adim(nu4)
        self.assertEqual(nu4, conv.compute_nu4_dim(nu4_adim))
