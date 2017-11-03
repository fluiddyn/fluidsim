import numpy as np


# pythran export qapamfft_from_uxuyetafft(
#     complex128[][], complex128[][], complex128[][],
#     int, int, float64[][], float64[][], float64[][],
#     complex128[][], float, float, int)
def qapamfft_from_uxuyetafft(
        ux_fft, uy_fft, eta_fft,
        n0, n1, KX, KY, K2,
        Kappa_over_ic, f, c2, rank):
    """Calculate normal modes from primitive variables."""
    freq_Corio = f
    f_over_c2 = freq_Corio / c2

    q_fft = np.empty([n0, n1], dtype=np.complex128)
    ap_fft = np.empty([n0, n1], dtype=np.complex128)
    am_fft = np.empty([n0, n1], dtype=np.complex128)

    if freq_Corio != 0:
        for i0 in range(n0):
            for i1 in range(n1):
                if i0 == 0 and i1 == 0 and rank == 0:
                    q_fft[i0, i1] = 0
                    ap_fft[i0, i1] = ux_fft[0, 0] + 1.j*uy_fft[0, 0]
                    am_fft[i0, i1] = ux_fft[0, 0] - 1.j*uy_fft[0, 0]
                else:

                    rot_fft = 1j*(
                        KX[i0, i1]*uy_fft[i0, i1] -
                        KY[i0, i1]*ux_fft[i0, i1])

                    q_fft[i0, i1] = rot_fft - freq_Corio*eta_fft[i0, i1]

                    a_over2_fft = 0.5*(
                        K2[i0, i1] * eta_fft[i0, i1] +
                        f_over_c2*rot_fft)

                    Deltaa_over2_fft = 0.5j*Kappa_over_ic[i0, i1]*(
                        KX[i0, i1]*ux_fft[i0, i1] +
                        KY[i0, i1]*uy_fft[i0, i1])

                    ap_fft[i0, i1] = a_over2_fft + Deltaa_over2_fft
                    am_fft[i0, i1] = a_over2_fft - Deltaa_over2_fft

    else:  # (freq_Corio == 0.)
        for i0 in range(n0):
            for i1 in range(n1):
                if i0 == 0 and i1 == 0 and rank == 0:
                    q_fft[i0, i1] = 0
                    ap_fft[i0, i1] = ux_fft[0, 0] + 1.j*uy_fft[0, 0]
                    am_fft[i0, i1] = ux_fft[0, 0] - 1.j*uy_fft[0, 0]
                else:
                    q_fft[i0, i1] = 1j*(
                        KX[i0, i1]*uy_fft[i0, i1] -
                        KY[i0, i1]*ux_fft[i0, i1])

                    a_over2_fft = 0.5*K2[i0, i1]*eta_fft[i0, i1]

                    Deltaa_over2_fft = 0.5j*Kappa_over_ic[i0, i1]*(
                        KX[i0, i1]*ux_fft[i0, i1] +
                        KY[i0, i1]*uy_fft[i0, i1])

                    ap_fft[i0, i1] = a_over2_fft + Deltaa_over2_fft
                    am_fft[i0, i1] = a_over2_fft - Deltaa_over2_fft

    return q_fft, ap_fft, am_fft
