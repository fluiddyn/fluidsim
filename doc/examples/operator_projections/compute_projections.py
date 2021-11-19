import numpy as np

import fluidsim.operators.operators3d as op
from fluidsim.operators.operators3d import OperatorsPseudoSpectral3D


p = OperatorsPseudoSpectral3D._create_default_params()

p.oper.nx = p.oper.ny = p.oper.nz = 100
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

# Projection along the k-radial direction
vx_fft_k, vy_fft_k, vz_fft_k = oper.project_kradial3d(vx_fft, vy_fft, vz_fft)
# Projection along the k-polar direction
vx_fft_p, vy_fft_p, vz_fft_p = oper.project_polar3d(vx_fft, vy_fft, vz_fft)
# Projection along the k-azimutal direction
vx_fft_a, vy_fft_a, vz_fft_a = oper.project_azim3d(vx_fft, vy_fft, vz_fft)

# Difference between the original field and the projection, called residual field
dvx_fft = vx_fft - vx_fft_k - vx_fft_p - vx_fft_a
dvy_fft = vy_fft - vy_fft_k - vy_fft_p - vy_fft_a
dvz_fft = vz_fft - vz_fft_k - vz_fft_p - vz_fft_a


# Energy contained in the original field
E_v = op.compute_energy_from_3fields(vx_fft, vy_fft, vz_fft)


# Energy contained in the difference between the original field and its projection
E_dv = op.compute_energy_from_3fields(dvx_fft, dvy_fft, dvz_fft)
assert np.max(
    E_dv / E_v < 1e-14
), "Problem in the projections: too much energy is in the residual field."


# Projection along the k-azimutal direction and then the k-polar direction
vx_fft_pa, vy_fft_pa, vz_fft_pa = oper.project_polar3d(
    vx_fft_a, vy_fft_a, vz_fft_a
)
E_pa = op.compute_energy_from_3fields(vx_fft_pa, vy_fft_pa, vz_fft_pa)
assert (
    np.max(E_pa / E_v) < 1e-14
), "Problem in the projections: too much energy is in the projection of an azimutal field on the polar direction."


# Projection along the k-radial direction and then the k-polar direction
vx_fft_pk, vy_fft_pk, vz_fft_pk = oper.project_polar3d(
    vx_fft_k, vy_fft_k, vz_fft_k
)
E_pk = op.compute_energy_from_3fields(vx_fft_pk, vy_fft_pk, vz_fft_pk)
assert (
    np.max(E_pk / E_v) < 1e-14
), "Problem in the projections: too much energy is in the projection of a radial field on the polar direction."


# Projection along the k-radial direction and then the k-azimutal direction
vx_fft_ak, vy_fft_ak, vz_fft_ak = oper.project_azim3d(
    vx_fft_k, vy_fft_k, vz_fft_k
)
E_ak = op.compute_energy_from_3fields(vx_fft_ak, vy_fft_ak, vz_fft_ak)
assert np.max(
    E_pk / E_v < 1e-14
), "Problem in the projections: too much energy is in the projection of a radial field on the azimutal direction."


# Here is a test to convice ourself that the computations are well done in the projections: we check that cos and sin are in [-1:1]
Kh_square = oper.Kx ** 2 + oper.Ky ** 2
K_square_nozero = Kh_square + oper.Kz ** 2
Kh_square_nozero = Kh_square.copy()

Kh_square_nozero[Kh_square_nozero == 0] = 1e-14
K_square_nozero[K_square_nozero == 0] = 1e-14

inv_Kh_square_nozero = 1.0 / Kh_square_nozero
inv_K_square_nozero = 1.0 / K_square_nozero

cos_theta_k = oper.Kz * np.sqrt(inv_K_square_nozero)
sin_theta_k = np.sqrt(Kh_square * inv_K_square_nozero)
cos_phi_k = oper.Kx * np.sqrt(inv_Kh_square_nozero)
sin_phi_k = oper.Ky * np.sqrt(inv_Kh_square_nozero)

assert (
    np.min(cos_theta_k) >= -1.0 and np.max(cos_theta_k) <= 1.0
), "cos_theta_k not in [-1:1]"
assert (
    np.min(sin_theta_k) >= -1.0 and np.max(cos_theta_k) <= 1.0
), "sin_theta_k not in [-1:1]"
assert (
    np.min(cos_phi_k) >= -1.0 and np.max(cos_theta_k) <= 1.0
), "cos_phi_k not in [-1:1]"
assert (
    np.min(sin_phi_k) >= -1.0 and np.max(cos_theta_k) <= 1.0
), "sin_phi_k not in [-1:1]"


print("Projections seems to be Ok.")
