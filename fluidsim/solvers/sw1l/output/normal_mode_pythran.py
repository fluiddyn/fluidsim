# pythran export get_qmat(
#     float, float, float64[][], float64[][], float64[][],
#     float64[][], float64[][], float64[][])
import numpy as np


def get_qmat(f, c, sigma, KX, KY, KK, K2, KK_not0):
    ck = c * KK_not0
    qmat = np.array(
        [[-1j * 2.**0.5 * ck * KY, +1j * f * KY + KX * sigma, +1j * f * KY - KX * sigma],
         [+1j * 2.**0.5 * ck * KX, -1j * f * KX + KY * sigma, -1j * f * KX - KY * sigma],
         [2.**0.5 * f * KK, c * K2, c * K2]]) / (2.**0.5 * sigma * KK_not0)
    return qmat
