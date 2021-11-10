import numpy as np

from fluidsim.operators.operators3d import OperatorsPseudoSpectral3D


p = OperatorsPseudoSpectral3D._create_default_params()

p.oper.nx = p.oper.ny = p.oper.nz = 100
lx = ly = lz = p.oper.Lx = p.oper.Ly = p.oper.Lz = 2 * np.pi

oper = OperatorsPseudoSpectral3D(params=p)

X, Y, Z = oper.get_XYZ_loc()

# A given velocity field
vx = np.sin(X / lx + 4 * Y / ly + 2 * Z / lz)
vy = 3.0 * np.sin(X / lx + 3 * Y / ly + 3 * Z / lz)
vz = 9.0 * np.sin(7 * X / lx + 2 * Y / ly + Z / lz)


vx_fft = oper.fft(vx)
vy_fft = oper.fft(vy)
vz_fft = oper.fft(vz)

# Projection along the k-radial direction
vx_fft_k = vz_fft.copy()
vy_fft_k = vy_fft.copy()
vz_fft_k = vz_fft.copy()
oper.project_perpk3d(vx_fft_k, vy_fft_k, vz_fft_k)
vx_fft_k, vy_fft_k, vz_fft_k = (
    vx_fft - vx_fft_k,
    vy_fft - vy_fft_k,
    vz_fft - vz_fft_k,
)
# Projection along the k-polar direction
a = oper.project_polar3d(vx_fft, vy_fft, vz_fft)
# Projection along the k-azimutal direction
# vx_fft_a, vy_fft_a, vz_fft_a = oper.project_azim3d(vx_fft, vy_fft, vz_fft)

print(a)

# print(vx_fft_p - vx_fft_a)

"""
print("Relative difference between vx_fft and the sum of its projections along ek, ek_theta, and ek_phi. Should be zero (up to numerical errors) if the projectors work well.")
print((vx_fft - vx_fft_k - vx_fft_p - vx_fft_a) / vx_fft)
print("Relative difference between vy_fft and the sum of its projections along ek, ek_theta, and ek_phi. Should be zero (up to numerical errors) if the projectors work well.")
print((vy_fft - vy_fft_k - vy_fft_p - vy_fft_a) / vy_fft)
print("Relative difference between vz_fft and the sum of its projections along ek, ek_theta, and ek_phi. Should be zero (up to numerical errors) if the projectors work well.")
print((vz_fft - vz_fft_k - vz_fft_p - vz_fft_a) / vz_fft)

print("Two consecutives projections along transversal direction should also be zero.")

# Projection along the k-azimutal direction and then the k-polar direction
vx_fft_pa = vx_fft_a.copy()
vy_fft_pa = vy_fft_a.copy()
vz_fft_pa = vz_fft_a.copy()
oper.project_polar3d(vx_fft_pa, vy_fft_pa, vz_fft_pa)
print(vx_fft_pa / vx_fft)
print(vy_fft_pa / vy_fft)
print(vz_fft_pa / vz_fft)

# Projection along the k-radial direction and then the k-polar direction
vx_fft_pk = vx_fft_k.copy()
vy_fft_pk = vy_fft_k.copy()
vz_fft_pk = vz_fft_k.copy()
oper.project_polar3d(vx_fft_pk, vy_fft_pk, vz_fft_pk)
print(vx_fft_pk / vx_fft)
print(vy_fft_pk / vy_fft)
print(vz_fft_pk / vz_fft)


# Projection along the k-radial direction and then the k-azimutal direction
vx_fft_ak = vx_fft_k.copy()
vy_fft_ak = vy_fft_k.copy()
vz_fft_ak = vz_fft_k.copy()
oper.project_azim3d(vx_fft_ak, vy_fft_ak, vz_fft_ak)
print(vx_fft_ak / vx_fft)
print(vy_fft_ak / vy_fft)
print(vz_fft_ak / vz_fft)
"""
