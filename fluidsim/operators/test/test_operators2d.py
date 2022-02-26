import numpy as np
from copy import deepcopy
import os
from math import prod

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

    if "FLUIDSIM_TYPE_FFT" in os.environ:
        type_fft = os.environ["FLUIDSIM_TYPE_FFT"]
        print(f"{type_fft = }")

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
class TestCoarse:
    nb_dim = 2

    @property
    def Oper(self):
        from fluidsim.operators.operators2d import OperatorsPseudoSpectral2D

        return OperatorsPseudoSpectral2D

    def test_coarse(self, allclose):

        params = self.Oper._create_default_params()
        params.oper.nx = 16
        params.oper.ny = 12
        if self.nb_dim == 3:
            params.oper.nz = 16

        params.oper.truncation_shape = "spherical"

        if "FLUIDSIM_TYPE_FFT" in os.environ:
            params.oper.type_fft = os.environ["FLUIDSIM_TYPE_FFT"]
            print(f"{params.oper.type_fft = }")

        oper = self.Oper(params)

        params_coarse = deepcopy(params)
        params_coarse.oper.nx = 4
        params_coarse.oper.ny = 4
        if self.nb_dim == 3:
            params_coarse.oper.nz = 4

        params_coarse.oper.type_fft = "sequential"
        params_coarse.oper.coef_dealiasing = 1.0

        if mpi.rank == 0:
            oper_coarse = oper.__class__(params=params_coarse)
            oper_coarse_shapeK = oper_coarse.shapeK_loc

            field_coarse = oper_coarse.create_arrayX_random()
            field_coarse_fft = oper_coarse.fft(field_coarse)

            # zeros because of conditions in put_coarse_array_in_array_fft
            # this corresponds to the largest values of kx, ky and kz
            if self.nb_dim == 2:
                nkyc, nkxc = oper_coarse_shapeK
                field_coarse_fft[nkyc // 2, :] = 0
                field_coarse_fft[:, nkxc - 1] = 0
            elif self.nb_dim == 3:
                nkzc, nkyc, nkxc = oper_coarse_shapeK
                field_coarse_fft[nkzc // 2, :, :] = 0
                field_coarse_fft[:, nkyc // 2, :] = 0
                field_coarse_fft[:, :, nkxc - 1] = 0

            field_coarse = oper_coarse.ifft(field_coarse_fft)

            energy = oper_coarse.compute_energy_from_X(field_coarse)
        else:
            oper_coarse = None
            oper_coarse_shapeK = None
            energy = None

        if mpi.nb_proc > 1:
            oper_coarse_shapeK = mpi.comm.bcast(oper_coarse_shapeK, root=0)
            energy = mpi.comm.bcast(energy, root=0)

        oper_coarse_sizeK = prod(oper_coarse_shapeK)

        if mpi.rank == 0:
            buffer = field_coarse_fft.flatten()
        else:
            buffer = np.empty(oper_coarse_sizeK, dtype=np.complex128)
        if mpi.nb_proc > 1:
            mpi.comm.Bcast(buffer, root=0)
        field_coarse_fft = buffer.reshape(oper_coarse_shapeK)

        print(f"{mpi.rank = }")
        print(f"{oper.shapeK_seq = }")
        print(f"{oper.shapeK_loc = }")
        if hasattr(oper, "dimX_K"):
            mpi.printby0(f"{oper.dimX_K = }")

        mpi.print_sorted(f"{oper_coarse_shapeK = }")
        mpi.print_sorted(f"{field_coarse_fft     =}")

        field_fft = oper.create_arrayK(value=0)
        oper.put_coarse_array_in_array_fft(
            field_coarse_fft, field_fft, oper_coarse, oper_coarse_shapeK
        )
        mpi.print_sorted(f"{field_fft            =}")

        energy_big_fft = oper.compute_energy_from_K(field_fft)
        mpi.printby0(
            "energy,  energy_big_fft\n"
            + (2 * "{:.8f}    ").format(energy, energy_big_fft)
        )
        assert energy > 0
        assert allclose(energy, energy_big_fft)

        field_coarse_fft_back = oper.coarse_seq_from_fft_loc(
            field_fft, oper_coarse_shapeK
        )

        mpi.printby0(f"{field_coarse_fft_back=}")

        if mpi.rank == 0:
            buffer = field_coarse_fft_back.flatten()
        else:
            buffer = np.empty(oper_coarse_sizeK, dtype=np.complex128)
        if mpi.nb_proc > 1:
            mpi.comm.Bcast(buffer, root=0)
        field_coarse_fft_back = buffer.reshape(oper_coarse_shapeK)

        assert allclose(field_coarse_fft.real, field_coarse_fft_back.real)
        assert allclose(field_coarse_fft.imag, field_coarse_fft_back.imag)
        # assert np.allclose(field_coarse_fft, field_coarse_fft_back)

        field = oper.ifft(field_fft)
        field_fft_back = oper.fft(field)
        # Test if field_fft corresponds to a real field
        assert allclose(field_fft.real, field_fft_back.real)
        assert allclose(field_fft.imag, field_fft_back.imag)
        # assert np.allclose(field_fft, field_fft_back)

        energy_big = oper.compute_energy_from_X(field)

        if mpi.rank == 0:
            assert allclose(energy, energy_big)

        if mpi.rank == 0:
            field_coarse_back = oper_coarse.ifft(field_coarse_fft_back)
            energy_back = oper_coarse.compute_energy_from_X(field_coarse_back)
        else:
            energy_back = None

        if mpi.nb_proc > 1:
            energy_back = mpi.comm.bcast(energy_back, root=0)

        mpi.printby0(
            "energy,  energy_back,  energy_big_fft,  energy_big\n"
            + (4 * "{:.8f}    ").format(
                energy, energy_back, energy_big_fft, energy_big
            )
        )
        assert allclose(energy, energy_back)
