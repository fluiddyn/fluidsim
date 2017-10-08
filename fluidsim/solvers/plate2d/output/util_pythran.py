
import numpy as np


# pythran export compute_correl4_seq(complex128[][], int32[], int, int)

def compute_correl4_seq(q_fftt, iomegas1, nb_omegas, nb_xs_seq):
    r"""Compute the correlations 4.

    .. math::
       C_4(\omega_1, \omega_2, \omega_3, \omega_4) =
       \langle
       \tilde w(\omega_1, \mathbf{x})
       \tilde w(\omega_2, \mathbf{x})
       \tilde w(\omega_3, \mathbf{x})^*
       \tilde w(\omega_4, \mathbf{x})^*
       \rangle_\mathbf{x},

    where

    .. math::
       \omega_2 = \omega_3 + \omega_4 - \omega_1

    and :math:`\omega_1 > 0`, :math:`\omega_3 > 0` and
    :math:`\omega_4 > 0`. Thus, this function produces an array
    :math:`C_4(\omega_1, \omega_3, \omega_4)`.

    """
    q_fftt_conj = np.conj(q_fftt)

    nx = q_fftt.shape[0]
    n0 = iomegas1.shape[0]

    corr4 = np.zeros((len(iomegas1), nb_omegas, nb_omegas),
                     dtype=np.complex128)

    for i1 in range(n0):
        io1 = iomegas1[i1]
        # this loop could be parallelized (OMP)
        for io3 in range(nb_omegas):
            # we use the symmetry omega_3 <--> omega_4
            for io4 in range(io3+1):
                io2 = io3 + io4 - io1
                if io2 < 0:
                    io2 = -io2
                    for ix in range(nx):
                        corr4[i1, io3, io4] += (
                            q_fftt[ix, io1] *
                            q_fftt_conj[ix, io3] *
                            q_fftt_conj[ix, io4] *
                            q_fftt_conj[ix, io2])
                elif io2 >= nb_omegas:
                    io2 = 2*nb_omegas-1-io2
                    for ix in range(nx):
                        corr4[i1, io3, io4] += (
                            q_fftt[ix, io1] *
                            q_fftt_conj[ix, io3] *
                            q_fftt_conj[ix, io4] *
                            q_fftt_conj[ix, io2])
                else:
                    for ix in range(nx):
                        corr4[i1, io3, io4] += (
                            q_fftt[ix, io1] *
                            q_fftt_conj[ix, io3] *
                            q_fftt_conj[ix, io4] *
                            q_fftt[ix, io2])
                # symmetry omega_3 <--> omega_4:
                corr4[i1, io4, io3] = corr4[i1, io3, io4]
    return corr4

# pythran export compute_correl2_seq(complex128[][], int32[], int, int)

def compute_correl2_seq(q_fftt, iomegas1, nb_omegas, nb_xs_seq):
    r"""Compute the correlations 2.

    .. math::
       C_2(\omega_1, \omega_2) =
       \langle
       \tilde w(\omega_1, \mathbf{x})
       \tilde w(\omega_2, \mathbf{x})^*
       \rangle_\mathbf{x},

    where :math:`\omega_1 = \omega_2`. Thus, this function
    produces an array :math:`C_2(\omega)`.

    """
    q_fftt_conj = np.conj(q_fftt)
    nx = q_fftt.shape[0]

    corr2 = np.zeros((nb_omegas, nb_omegas), dtype=np.complex128)

    for io3 in range(nb_omegas):
        for io4 in range(io3+1):
            for ix in range(nx):
                corr2[io3, io4] += (
                    q_fftt[ix, io3] * q_fftt_conj[ix, io4])
            corr2[io4, io3] = np.conj(corr2[io3, io4])

    return corr2
