import unittest
import numpy as np
import sys
from copy import deepcopy

import fluiddyn.util.mpi as mpi

from fluidsim.util.testing import TestCase, skip_if_no_fluidfft


def create_oper(type_fft=None, coef_dealiasing=2.0 / 3, **kwargs):
    from fluidsim.operators.operators2d import OperatorsPseudoSpectral2D

    params = OperatorsPseudoSpectral2D._create_default_params()

    params.ONLY_COARSE_OPER = kwargs.get("ONLY_COARSE_OPER", False)

    params._set_attrib("f", 0)
    params._set_attrib("c2", 100)
    params._set_attrib("kd2", 0)

    if "nh" in kwargs:
        nh = kwargs["nh"]
    elif mpi.nb_proc == 1:
        nh = 9
    else:
        nh = 8

    params.oper.nx = nh
    params.oper.ny = nh
    Lh = 6.0
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    if type_fft is not None:
        params.oper.type_fft = type_fft

    params.oper.coef_dealiasing = coef_dealiasing

    oper = OperatorsPseudoSpectral2D(params=params)

    return oper


def compute_increments_dim1_old(var, irx):
    """Old version of the function compute_increments_dim1."""
    n0 = var.shape[0]
    n1 = var.shape[1]
    n1new = n1 - irx
    inc_var = np.empty([n0, n1new])
    for i0 in range(n0):
        for i1 in range(n1new):
            inc_var[i0, i1] = var[i0, i1 + irx] - var[i0, i1]
    return inc_var


@skip_if_no_fluidfft
@unittest.skipIf(sys.platform.startswith("win"), "Untested on Windows")
class TestOperators(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oper = create_oper()
        cls.rtol = 1e-15
        cls.atol = 1e-14  # Absolute tolerance for double precision FFT

    def test_curl(self):
        """Test curl"""
        oper = self.oper
        rot = oper.create_arrayX_random()
        rot_fft = oper.fft2(rot)
        if mpi.rank == 0:
            rot_fft[0, 0] = 0.0

        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
        rot2_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)

        np.testing.assert_allclose(rot2_fft, rot_fft, self.rtol, self.atol)

    def test_laplacian(self):
        oper = self.oper
        ff = oper.create_arrayX_random()
        ff_fft = oper.fft2(ff)
        if mpi.rank == 0:
            ff_fft[0, 0] = 0.0

        lap_fft = oper.laplacian_fft(ff_fft, order=4)
        ff_fft_back = oper.invlaplacian_fft(lap_fft, order=4)

        np.testing.assert_allclose(ff_fft, ff_fft_back, self.rtol, self.atol)

        lap_fft = oper.laplacian_fft(ff_fft, order=2, negative=True)
        invlap_fft = oper.invlaplacian_fft(ff_fft, order=2, negative=True)

        np.testing.assert_equal(oper.K2 * ff_fft, lap_fft)
        np.testing.assert_allclose(
            ff_fft / oper.K2_not0, invlap_fft, self.rtol, self.atol
        )

    def test_compute_increments_dim1(self):
        """Test computing increments of var over the dim 1."""
        oper = self.oper
        var = oper.create_arrayX_random()

        def assert_increments_equal(irx):
            inc_var = oper.compute_increments_dim1(var, irx)
            inc_var_old = compute_increments_dim1_old(var, irx)
            np.testing.assert_equal(inc_var_old, inc_var)

        n1 = oper.shapeX[1]
        for irx in [n1, n1 // 2, 0]:
            assert_increments_equal(irx)


@skip_if_no_fluidfft
class TestOperatorOnlyCoarse(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nh = 12
        cls.oper = create_oper(ONLY_COARSE_OPER=True, nh=cls.nh)

    def test_oper_coarse(self):
        """Test coarse operator parameters which, by default, initializes
        `nh=4`.

        """
        oper = self.oper

        # Assert params are intact but the operator is initialized coarse
        self.assertEqual(oper.params.oper.nx, self.nh)
        self.assertEqual(oper.params.oper.ny, self.nh)
        self.assertNotEqual(oper.nx, self.nh)
        self.assertNotEqual(oper.ny, self.nh)


@skip_if_no_fluidfft
class TestOperatorsDealiasing(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oper = create_oper(coef_dealiasing=False)

    def test_dealiasing(self):
        """Test if dealiasing with coef_dealiasing=1.0 keeps the original
        array unchanged.

        """
        oper = self.oper
        var_fft = oper.create_arrayK(1.0)
        sum_var = var_fft.sum()
        oper.dealiasing(var_fft)
        sum_var_dealiased = var_fft.sum()
        self.assertEqual(sum_var, sum_var_dealiased)


@skip_if_no_fluidfft
class TestCoarse(unittest.TestCase):
    nb_dim = 2

    @property
    def Oper(self):
        from fluidsim.operators.operators2d import OperatorsPseudoSpectral2D

        return OperatorsPseudoSpectral2D

    def test_coarse(self):

        params = self.Oper._create_default_params()
        params.oper.nx = 32
        params.oper.ny = 48
        if self.nb_dim == 3:
            params.oper.nz = 12

        params.oper.truncation_shape = "spherical"

        oper = self.Oper(params)

        params_coarse = deepcopy(params)
        params_coarse.oper.nx = 2 * 8
        params_coarse.oper.ny = 2 * 12
        if self.nb_dim == 3:
            params_coarse.oper.nz = 2 * 4

        params_coarse.oper.type_fft = "sequential"
        params_coarse.oper.coef_dealiasing = 1.0

        if mpi.rank == 0:
            oper_coarse = oper.__class__(params=params_coarse)
            oper_coarse_shapeK_loc = oper_coarse.shapeK_loc

            field_coarse = oper_coarse.create_arrayX_random()
            field_coarse_fft = oper_coarse.fft(field_coarse)

            if self.nb_dim == 2:

                nkyc, nkxc = oper_coarse_shapeK_loc
                # zeros because of conditions in put_coarse_array_in_array_fft
                field_coarse_fft[nkyc // 2, :] = 0
                field_coarse_fft[:, nkxc - 1] = 0
            elif self.nb_dim == 3:
                nkzc, nkyc, nkxc = oper_coarse_shapeK_loc
                field_coarse_fft[nkzc // 2, :] = 0
                field_coarse_fft[:, nkyc // 2, :] = 0
                field_coarse_fft[:, :, nkxc - 1] = 0

            field_coarse = oper_coarse.ifft(field_coarse_fft)
            energy = oper_coarse.compute_energy_from_X(field_coarse)
        else:
            field_coarse_fft = None
            oper_coarse = None
            oper_coarse_shapeK_loc = None

        if mpi.nb_proc > 1:
            oper_coarse_shapeK_loc = mpi.comm.bcast(
                oper_coarse_shapeK_loc, root=0
            )

        field_fft = oper.create_arrayK(value=0)
        oper.put_coarse_array_in_array_fft(
            field_coarse_fft, field_fft, oper_coarse, oper_coarse_shapeK_loc
        )

        energy_big_fft = oper.compute_energy_from_K(field_fft)

        if mpi.rank == 0:
            print(
                "energy,  energy_big_fft\n"
                + (2 * "{:.8f}    ").format(energy, energy_big_fft)
            )
            assert energy > 0
            assert np.allclose(energy, energy_big_fft)
            print("OK np.allclose(energy, energy_big_fft)")

        field = oper.ifft(field_fft)
        energy_big = oper.compute_energy_from_X(field)

        field_fft_back = oper.fft(field)
        if not np.allclose(field_fft, field_fft_back):
            print("Buggy! field_fft != field_fft_back")

        field_coarse_fft_back = oper.coarse_seq_from_fft_loc(
            field_fft, oper_coarse_shapeK_loc
        )

        if mpi.rank == 0:
            field_coarse_back = oper_coarse.ifft(field_coarse_fft_back)
            energy_back = oper_coarse.compute_energy_from_X(field_coarse_back)
            print(
                "energy,  energy_back,  energy_big_fft,  energy_big\n"
                + (4 * "{:.8f}    ").format(
                    energy, energy_back, energy_big_fft, energy_big
                )
            )
            assert np.allclose(energy, energy_back)
            print("OK np.allclose(energy, energy_back)")
            assert np.allclose(energy, energy_big)


if __name__ == "__main__":
    unittest.main()
