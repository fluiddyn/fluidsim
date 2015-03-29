"""
Correl freq (:mod:`fluidsim.solvers.plate2d.output.correlations_freq`)
============================================================================

.. currentmodule:: fluidsim.solvers.plate2d.output.correlations_freq

Provides:

.. autoclass:: CorrelationsFreq
   :members:
   :private-members:

"""
import h5py

import numpy as np

# from fluiddyn.util import mpi

from .correlations_freq import CorrelationsFreq as CorrelationsFreqPython


class CorrelationsFreq(CorrelationsFreqPython):
    """Compute, save, load and plot correlations of frequencies.

    """

    def _compute_correl4(self, w):
        r"""Compute the correlations 4.

        .. math::
           C_4(\omega_1, \omega_2, \omega_3, \omega_4) =
           \langle
           \tilde w(\omega_1, \mathbf{x}) +
           \tilde w(\omega_2, \mathbf{x}) +
           \tilde w(\omega_3, \mathbf{x})^* +
           \tilde w(\omega_4, \mathbf{x})^* +
           \rangle_\mathbf{x},

        where

        .. math::
           \omega_2 = \omega_3 + \omega_4 - \omega_1

        and :math:`\omega_1 > 0`, :math:`\omega_3 > 0` and
        :math:`\omega_4 > 0`. Thus, this function produces an array
        :math:`C_4(\omega_1, \omega_3, \omega_4)`.

        """

        nt, ny, nx = w.shape

        w_fftt = self.oper_fft1.fft(w).reshape([nt, nx*ny])
        w_fftt_conj = w_fftt.conj()

        nb_omegas = self.nb_omegas

        iomegas1 = self.iomegas1

        corr4 = np.empty([len(iomegas1), nb_omegas, nb_omegas])

        for i1, io1 in enumerate(iomegas1):
            # this loop could be parallelized (OMP)
            for io3 in range(nb_omegas):
                # we use the symmetry omega_3 <--> omega_4
                for io4 in range(0, io3+1):
                    tmp = (w_fftt[io1, :] *
                           w_fftt_conj[io3, :] *
                           w_fftt_conj[io4, :])
                    io2 = io3 + io4 - io1
                    if io2 < 0:
                        io2 = abs(io2)
                        corr4[i1, io3, io4] = np.mean(tmp*w_fftt_conj[io2, :])
                    else:
                        corr4[i1, io3, io4] = np.mean(tmp*w_fftt[io3, :])
                # symmetry omega_3 <--> omega_4:
                corr4[i1, io4, io3] = corr4[i1, io3, io4]

        # if mpi.nb_proc > 1:
        #     # reduce for mean:
        #     mpi.comm.

        return corr4
