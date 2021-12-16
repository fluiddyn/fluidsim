import unittest

import numpy as np

from .test_operators2d import TestCoarse as _TestCoarse


class TestCoarse(_TestCoarse):
    nb_dim = 3

    @property
    def Oper(self):
        from fluidsim.operators.operators3d import OperatorsPseudoSpectral3D

        return OperatorsPseudoSpectral3D


del _TestCoarse


def test_projection():
    from fluidsim.operators.operators3d import (
        OperatorsPseudoSpectral3D,
        compute_energy_from_3fields,
        compute_energy_from_1field,
    )

    p = OperatorsPseudoSpectral3D._create_default_params()

    p.oper.nx = p.oper.ny = p.oper.nz = 16
    lx = ly = lz = p.oper.Lx = p.oper.Ly = p.oper.Lz = 2 * np.pi

    oper = OperatorsPseudoSpectral3D(params=p)

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
    assert np.sum(E_t) > 0.1 * np.sum(E_v)
    dE_t = E_t_s - E_t
    assert np.max(
        dE_t / E_v < 1e-14
    ), "Too much energy difference in the toroidal projections done with vtfft_from_vecfft and project_toroidal."

    # Test of vecfft_from_vpfft
    # Recompute the velocity field corresponding to the poloidal projection vp_fft
    vx_fft_p, vy_fft_p, vz_fft_p = oper.vecfft_from_vpfft(vp_fft)
    E_p = compute_energy_from_3fields(vx_fft_p, vy_fft_p, vz_fft_p)
    assert np.sum(E_p) > 0.1 * np.sum(E_v)

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

    print("Projections seems to be Ok.")


if __name__ == "__main__":
    unittest.main()
