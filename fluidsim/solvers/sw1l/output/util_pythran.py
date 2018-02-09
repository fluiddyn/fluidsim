import numpy as np


# pythran export get_qmat(
#     float or int, float or int, float64[][], float64[][], float64[][],
#     float64[][], float64[][], float64[][])
def get_qmat(f, c, sigma, KX, KY, KK, K2, KK_not0):
    """Compute Q matrix to transform q, ap, am (fft) -> b0, b+, b- (fft) with
    dimensions of velocity.

    """
    ck = c * KK_not0
    qmat = np.array(
        [[-1j * 2.**0.5 * ck * KY, +1j * f * KY + KX * sigma,
          +1j * f * KY - KX * sigma],
         [+1j * 2.**0.5 * ck * KX, -1j * f * KX + KY * sigma,
          -1j * f * KX - KY * sigma],
         [2.**0.5 * f * KK, c * K2, c * K2]]) / (2.**0.5 * sigma * KK_not0)
    return qmat


# pythran export linear_eigenmode_from_values_1k(
#     complex128, complex128, complex128, float, float,
#     float or int, float or int)
def linear_eigenmode_from_values_1k(
        ux_fft, uy_fft, eta_fft, kx, ky, f, c2):
    """Compute q, d, a (fft) for a single wavenumber."""
    div_fft = 1j * (kx * ux_fft + ky * uy_fft)
    rot_fft = 1j * (kx * uy_fft - ky * ux_fft)
    q_fft = rot_fft - f * eta_fft
    k2 = kx ** 2 + ky ** 2
    ageo_fft = f * rot_fft / c2 + k2 * eta_fft
    return q_fft, div_fft, ageo_fft
