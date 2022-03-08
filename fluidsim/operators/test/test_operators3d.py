from math import tau
import os
from copy import deepcopy

import pytest

import numpy as np

from fluiddyn.util import mpi

from fluidsim.util.test_util import skip_if_no_fluidfft
from fluidsim.util.testing import FLUIDFFT_INSTALLED
from .test_operators2d import TestCoarse as _TestCoarse


def xfail_if_fluidfft_class_not_importable(func):
    if not FLUIDFFT_INSTALLED or "FLUIDSIM_TYPE_FFT" not in os.environ:
        return func

    from fluidfft.fft3d import import_fft_class

    try:
        import_fft_class(os.environ["FLUIDSIM_TYPE_FFT"])
    except ImportError:
        ImportError_fft_class = True
    else:
        ImportError_fft_class = False

    return pytest.mark.xfail(
        ImportError_fft_class, reason="FluidFFT class can't be imported"
    )(func)


@pytest.fixture(scope="module")
def oper():
    from fluidsim.operators.operators3d import OperatorsPseudoSpectral3D

    p = OperatorsPseudoSpectral3D._create_default_params()
    p.oper.nx = 16
    p.oper.ny = 11
    p.oper.nz = 4
    p.oper.Lx = p.oper.Ly = p.oper.Lz = 2 * np.pi

    if "FLUIDSIM_TYPE_FFT" in os.environ:
        p.oper.type_fft = os.environ["FLUIDSIM_TYPE_FFT"]
    print(f"{p.oper.type_fft = }")

    return OperatorsPseudoSpectral3D(params=p)


@xfail_if_fluidfft_class_not_importable
@skip_if_no_fluidfft
def test_projection(oper, allclose):
    from fluidsim.operators.operators3d import (
        compute_energy_from_3fields,
        compute_energy_from_1field,
    )

    lx = ly = lz = oper.Lx = oper.Ly = oper.Lz
    X, Y, Z = oper.get_XYZ_loc()

    # A given velocity field
    vx = 60.0 * np.sin(X / lx + 4 * Y / ly + 2 * Z / lz)
    vy = 30.0 * np.sin(X / lx + 3 * Y / ly + 3 * Z / lz)
    vz = 96.0 * np.sin(7 * X / lx + 2 * Y / ly + Z / lz)

    vx_fft = oper.fft(vx)
    vy_fft = oper.fft(vy)
    vz_fft = oper.fft(vz)

    vx_fft_k = vx_fft.copy()
    vy_fft_k = vy_fft.copy()
    vz_fft_k = vz_fft.copy()

    vx_fft_p = vx_fft.copy()
    vy_fft_p = vy_fft.copy()
    vz_fft_p = vz_fft.copy()

    vx_fft_t = vx_fft.copy()
    vy_fft_t = vy_fft.copy()
    vz_fft_t = vz_fft.copy()

    # Projection along the k-radial direction
    oper.project_kradial3d(vx_fft_k, vy_fft_k, vz_fft_k)
    # Projection along the poloidal direction
    oper.project_poloidal(vx_fft_p, vy_fft_p, vz_fft_p)
    # Projection along the toroidal direction
    oper.project_toroidal(vx_fft_t, vy_fft_t, vz_fft_t)

    # Difference between the original field and the projection, called residual field
    dvx_fft = vx_fft - vx_fft_k - vx_fft_p - vx_fft_t
    dvy_fft = vy_fft - vy_fft_k - vy_fft_p - vy_fft_t
    dvz_fft = vz_fft - vz_fft_k - vz_fft_p - vz_fft_t

    # Energy contained in the original field
    E_v = compute_energy_from_3fields(vx_fft, vy_fft, vz_fft)

    # Energy contained in the difference between the original field and its projection
    E_dv = compute_energy_from_3fields(dvx_fft, dvy_fft, dvz_fft)

    assert np.max(E_dv / E_v < 1e-14), "Too much energy is in the residual field."

    # Projection along the toroidal direction and then the poloidal direction
    vx_fft_pt = vx_fft_t.copy()
    vy_fft_pt = vy_fft_t.copy()
    vz_fft_pt = vz_fft_t.copy()
    oper.project_poloidal(vx_fft_pt, vy_fft_pt, vz_fft_pt)
    E_pt = compute_energy_from_3fields(vx_fft_pt, vy_fft_pt, vz_fft_pt)
    assert (
        np.max(E_pt / E_v) < 1e-14
    ), "Too much energy is in the poloidal projection of the toroidal field."

    # Projection along the k-radial direction and then the poloidal direction
    vx_fft_pk = vx_fft_k.copy()
    vy_fft_pk = vy_fft_k.copy()
    vz_fft_pk = vz_fft_k.copy()
    oper.project_poloidal(vx_fft_pk, vy_fft_pk, vz_fft_pk)
    E_pk = compute_energy_from_3fields(vx_fft_pk, vy_fft_pk, vz_fft_pk)
    assert (
        np.max(E_pk / E_v) < 1e-14
    ), "Too much energy is in the poloidal projection of the k-radial field."

    # Projection along the k-radial direction and then the toroidal direction
    vx_fft_tk = vx_fft_k.copy()
    vy_fft_tk = vy_fft_k.copy()
    vz_fft_tk = vz_fft_k.copy()
    oper.project_toroidal(vx_fft_tk, vy_fft_tk, vz_fft_tk)
    E_tk = compute_energy_from_3fields(vx_fft_tk, vy_fft_tk, vz_fft_tk)
    assert np.max(
        E_tk / E_v < 1e-14
    ), "Too much energy is in the toroidal projection of the k-radial field."

    # Test of vpfft_from_vecfft
    # Compute projection along the poloidal direction
    vp_fft = oper.vpfft_from_vecfft(vx_fft, vy_fft, vz_fft)
    E_p_s = compute_energy_from_1field(vp_fft)
    E_p = compute_energy_from_3fields(vx_fft_p, vy_fft_p, vz_fft_p)
    dE_p = E_p_s - E_p
    assert np.max(
        dE_p / E_v < 1e-14
    ), "Too much energy difference in the poloidal projections done with vpfft_from_vecfft and project_poloidal."

    # Test of vtfft_from_vecfft
    # Compute projection along the toroidal direction
    vt_fft = oper.vtfft_from_vecfft(vx_fft, vy_fft, vz_fft)
    E_t_s = compute_energy_from_1field(vt_fft)
    E_t = compute_energy_from_3fields(vx_fft_t, vy_fft_t, vz_fft_t)
    assert np.sum(E_t) > 0.05 * np.sum(E_v)
    dE_t = E_t_s - E_t
    assert np.max(
        dE_t / E_v < 1e-14
    ), "Too much energy difference in the toroidal projections done with vtfft_from_vecfft and project_toroidal."

    # Test of vecfft_from_vpfft
    # Recompute the velocity field corresponding to the poloidal projection vp_fft
    vx_fft_p, vy_fft_p, vz_fft_p = oper.vecfft_from_vpfft(vp_fft)
    E_p = compute_energy_from_3fields(vx_fft_p, vy_fft_p, vz_fft_p)
    assert np.sum(E_p) > 0.05 * np.sum(E_v)

    dE_p = E_p_s - E_p
    assert np.max(
        dE_p / E_v < 1e-14
    ), "Too much energy difference in the poloidal velocity fields computed with project_poloidal and vecfft_from_vpfft."

    # Test of vecfft_from_vtfft
    # Recompute the velocity field corresponding to the toroidal projection vt_fft
    vx_fft_t, vy_fft_t, vz_fft_t = oper.vecfft_from_vtfft(vt_fft)
    E_t = compute_energy_from_3fields(vx_fft_t, vy_fft_t, vz_fft_t)
    dE_t = E_t_s - E_t
    assert np.max(
        dE_t / E_v < 1e-14
    ), "Too much energy difference in the toroidal velocity fields computed with project_toroidal and vecfft_from_vtfft."

    def allclose_c(a, b):
        allclose(a.real, b.real)
        allclose(a.imag, b.imag)

    allclose_c(vx_fft_t, vx_fft_pt)
    allclose_c(vy_fft_t, vy_fft_pt)
    allclose_c(vz_fft_t, vz_fft_pt)


@xfail_if_fluidfft_class_not_importable
@skip_if_no_fluidfft
def test_divh_rotz(oper):
    lx = ly = oper.Lx = oper.Ly = oper.Lz
    X, Y, _ = oper.get_XYZ_loc()

    vxd = np.sin(tau * X / lx)
    vyd = 1.2345 * np.sin(tau * Y / ly)
    vxr = np.sin(tau * Y / ly)
    vyr = -np.sin(tau * X / lx)

    vx = vxd + vxr
    vy = vyd + vyr

    vx_fft = oper.fft(vx)
    vy_fft = oper.fft(vy)

    if mpi.rank == 0:
        vx_fft[0, 0, 0] = 0
        vy_fft[0, 0, 0] = 0

    divh_fft = oper.divhfft_from_vxvyfft(vx_fft, vy_fft)
    rotz_fft = oper.rotzfft_from_vxvyfft(vx_fft, vy_fft)

    vxd_fft, vyd_fft = oper.vxvyfft_from_divhfft(divh_fft)
    vxr_fft, vyr_fft = oper.vxvyfft_from_rotzfft(rotz_fft)

    assert np.allclose(divh_fft, oper.divhfft_from_vxvyfft(vxd_fft, vyd_fft))
    assert np.allclose(rotz_fft, oper.rotzfft_from_vxvyfft(vxr_fft, vyr_fft))
    assert np.allclose(vx_fft, vxd_fft + vxr_fft)
    assert np.allclose(vy_fft, vyd_fft + vyr_fft)
    assert np.allclose(vxd_fft, oper.fft(vxd))
    assert np.allclose(vxr_fft, oper.fft(vxr))
    assert np.allclose(vyd_fft, oper.fft(vyd))
    assert np.allclose(vyr_fft, oper.fft(vyr))

    if mpi.nb_proc > 1:
        return

    divh = oper.ifft(divh_fft)
    assert divh[0, oper.ny // 2, oper.nx // 2] < 0.0


@xfail_if_fluidfft_class_not_importable
@skip_if_no_fluidfft
def test_where_is_wavenumber(oper):
    from fluidsim.operators.operators3d import _ik_from_ikc

    nkx_seq, nky_seq, nkz_seq = oper.ixyz_from_i012(*oper.shapeK_seq)

    params_coarse = deepcopy(oper.params)
    params_coarse.oper.type_fft = "sequential"
    params_coarse.oper.coef_dealiasing = 1.0

    params_coarse.oper.nx = oper.params.oper.nx // 4
    params_coarse.oper.ny = oper.params.oper.ny // 8
    params_coarse.oper.nz = oper.params.oper.nz // 2

    oper_coarse = oper.__class__(params=params_coarse)
    assert oper_coarse.oper_fft.get_dimX_K() == (0, 1, 2)
    nkzc, nkyc, nkxc = oper_coarse.shapeK

    sendbuf = np.empty(4, dtype="i")
    recvbuf = None
    if mpi.rank == 0:
        recvbuf = np.empty([mpi.nb_proc, 4], dtype="i")

    print(f"{oper.shapeK_seq = }")
    print(f"{oper.shapeK_loc = }")
    if hasattr(oper, "dimX_K"):
        mpi.printby0(f"{oper.dimX_K = }")

    print(f"{oper.SAME_SIZE_IN_ALL_PROC = }")

    # print(f"{oper_coarse.Kx[0, 0] = }")
    # print(f"{oper.Kx = }")

    for ikcz in range(nkzc):
        for ikcy in range(nkyc):
            for ikcx in range(nkxc):
                ikz = _ik_from_ikc(ikcz, nkzc, nkz_seq)
                iky = _ik_from_ikc(ikcy, nkyc, nky_seq)
                ikx = ikcx

                ik0, ik1, ik2 = oper.i012_from_ixyz(ikx, iky, ikz)
                rank_k, ik0_loc, ik1_loc, ik2_loc = oper.where_is_wavenumber(
                    ik0, ik1, ik2
                )
                print(
                    f"{(ik0, ik1, ik2) = } => {(rank_k, ik0_loc, ik1_loc, ik2_loc) = }"
                )

                if mpi.nb_proc == 1:
                    assert rank_k == 0
                else:
                    sendbuf[:] = (rank_k, ik0_loc, ik1_loc, ik2_loc)
                    mpi.comm.Gather(sendbuf, recvbuf, root=0)
                    if mpi.rank == 0:
                        assert abs(np.diff(recvbuf, axis=0)).max() == 0

                kxc = oper_coarse.Kx[ikcz, ikcy, ikcx]
                kyc = oper_coarse.Ky[ikcz, ikcy, ikcx]
                kzc = oper_coarse.Kz[ikcz, ikcy, ikcx]

                if mpi.rank == rank_k:
                    kx = oper.Kx[ik0_loc, ik1_loc, ik2_loc]
                    ky = oper.Ky[ik0_loc, ik1_loc, ik2_loc]
                    kz = oper.Kz[ik0_loc, ik1_loc, ik2_loc]
                    data = np.array([kx, ky, kz])
                else:
                    data = np.empty(3, dtype=np.float64)

                if mpi.nb_proc > 1:
                    mpi.comm.Bcast(data, root=rank_k)

                kx, ky, kz = data
                assert np.allclose(kxc, kx)
                assert np.allclose(kyc, ky)
                assert np.allclose(kzc, kz)


@xfail_if_fluidfft_class_not_importable
class TestCoarse(_TestCoarse):
    nb_dim = 3

    @property
    def Oper(self):
        from fluidsim.operators.operators3d import OperatorsPseudoSpectral3D

        return OperatorsPseudoSpectral3D


del _TestCoarse
