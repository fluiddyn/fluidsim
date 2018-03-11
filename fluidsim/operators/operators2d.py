"""Operators 2d (:mod:`fluidsim.operators.operators2d`)
=======================================================

Provides

.. autoclass:: OperatorsPseudoSpectral2D
   :members:
   :private-members:

"""

from builtins import range

import numpy as np

from fluiddyn.util import mpi

from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D as _Operators

from . import util2d_pythran
from .util2d_pythran import (
    dealiasing_setofvar, laplacian2_fft, invlaplacian2_fft,
    compute_increments_dim1)
from ..base.setofvariables import SetOfVariables

if not hasattr(util2d_pythran, '__pythran__'):
    import warnings
    warnings.warn('util2d_pythran has to be pythranized to be efficient! '
                  'Install pythran and recompile.')

nb_proc = mpi.nb_proc
rank = mpi.rank
if nb_proc > 1:
    MPI = mpi.MPI
    comm = mpi.comm


class OperatorsPseudoSpectral2D(_Operators):

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        if nb_proc > 1:
            type_fft = 'fft2d.mpi_with_fftw1d'
        else:
            type_fft = 'fft2d.with_pyfftw'

        attribs = {'type_fft': type_fft,
                   'coef_dealiasing': 2./3,
                   'nx': 48,
                   'ny': 48,
                   'Lx': 8,
                   'Ly': 8}
        params._set_child('oper', attribs=attribs)

    def __init__(self, params, SEQUENTIAL=None, goal_to_print=None):

        self.params = params

        super(OperatorsPseudoSpectral2D, self).__init__(
            params.oper.nx, params.oper.ny, params.oper.Lx, params.oper.Ly,
            fft=params.oper.type_fft,
            coef_dealiasing=params.oper.coef_dealiasing)

        self.Lx = self.lx
        self.Ly = self.ly

        try:
            self.project_fft_on_realX = self._opfft.project_fft_on_realX
        except AttributeError:
            if self.is_sequential:
                self.project_fft_on_realX = self.project_fft_on_realX_seq
            else:
                self.project_fft_on_realX = self.project_fft_on_realX_slow

        if not self.is_sequential:

            self.iKxloc_start, _ = self.opfft.get_seq_indices_first_K()
            self.iKxloc_start_rank = np.array(
                comm.allgather(self.iKxloc_start))

            nkx_loc_rank = np.array(comm.allgather(self.nkx_loc))
            a = nkx_loc_rank
            self.SAME_SIZE_IN_ALL_PROC = (a >= a.max()).all()

        try:
            # for shallow water models
            self.Kappa2 = self.K2 + self.params.kd2
            self.Kappa_over_ic = -1.j * np.sqrt(self.Kappa2/self.params.c2)
            if self.params.f != 0:
                self.f_over_c2Kappa2 = self.params.f/(
                    self.params.c2*self.Kappa2)

        except AttributeError:
            pass

    def dealiasing(self, *args):
        for thing in args:
            if isinstance(thing, SetOfVariables):
                dealiasing_setofvar(thing, self.where_dealiased,
                                    self.nK0_loc, self.nK1_loc)
            elif isinstance(thing, np.ndarray):
                self.dealiasing_variable(thing)

    def dealiasing_setofvar(self, sov):
        dealiasing_setofvar(sov, self.where_dealiased,
                            self.nK0_loc, self.nK1_loc)

    def project_fft_on_realX_seq(self, f_fft):
        """Project the given field in spectral space such as its
        inverse fft is a real field."""

        nky_seq = self.shapeK_seq[0]

        iky_ky0 = 0
        iky_kyM = nky_seq//2
        ikx_kx0 = 0
        # ikx_kxM = self.nkx_seq-1
        ikx_kxM = self.shapeK_seq[1]-1

        # first, some values have to be real
        f_fft[iky_ky0, ikx_kx0] = f_fft[iky_ky0, ikx_kx0].real
        f_fft[iky_ky0, ikx_kxM] = f_fft[iky_ky0, ikx_kxM].real
        f_fft[iky_kyM, ikx_kx0] = f_fft[iky_kyM, ikx_kx0].real
        f_fft[iky_kyM, ikx_kxM] = f_fft[iky_kyM, ikx_kxM].real

        # second, there are relations between some values
        for ikyp in range(1, iky_kyM):
            ikyn = nky_seq - ikyp

            f_kp_kx0 = f_fft[ikyp, ikx_kx0]
            f_kn_kx0 = f_fft[ikyn, ikx_kx0]

            f_fft[ikyp, ikx_kx0] = (f_kp_kx0+f_kn_kx0.conjugate()
                                    )/2
            f_fft[ikyn, ikx_kx0] = ((f_kp_kx0+f_kn_kx0.conjugate()
                                     )/2).conjugate()

            f_kp_kxM = f_fft[ikyp, ikx_kxM]
            f_kn_kxM = f_fft[ikyn, ikx_kxM]

            f_fft[ikyp, ikx_kxM] = (f_kp_kxM+f_kn_kxM.conjugate()
                                    )/2
            f_fft[ikyn, ikx_kxM] = ((f_kp_kxM+f_kn_kxM.conjugate()
                                     )/2).conjugate()
        return f_fft

    def project_fft_on_realX_slow(self, f_fft):
        return self.fft(self.ifft(f_fft))

    def coarse_seq_from_fft_loc(self, f_fft, shapeK_loc_coarse):
        """Return a coarse field in K space."""
        nKyc = shapeK_loc_coarse[0]
        nKxc = shapeK_loc_coarse[1]

        if nb_proc > 1:
            if not self.is_transposed:
                raise NotImplementedError()

            fc_trans = np.empty([nKxc, nKyc], np.complex128)
            nKy = self.shapeK_seq[1]
            f1d_temp = np.empty([nKyc], np.complex128)

            for iKxc in range(nKxc):
                kx = self.deltakx*iKxc
                rank_iKx, iKxloc, iKyloc = self.where_is_wavenumber(kx, 0.)
                if rank == rank_iKx:
                    # create f1d_temp
                    for iKyc in range(nKyc):
                        if iKyc <= nKyc/2:
                            iKy = iKyc
                        else:
                            kynodim = iKyc - nKyc
                            iKy = kynodim + nKy
                        f1d_temp[iKyc] = f_fft[iKxloc, iKy]

                if rank_iKx != 0:
                    # message f1d_temp
                    if rank == 0:
                        # print('f1d_temp', f1d_temp, f1d_temp.dtype)
                        comm.Recv(
                            [f1d_temp, MPI.DOUBLE_COMPLEX],
                            source=rank_iKx, tag=iKxc)
                    elif rank == rank_iKx:
                        comm.Send(
                            [f1d_temp, MPI.DOUBLE_COMPLEX],
                            dest=0, tag=iKxc)
                if rank == 0:
                    # copy into fc_trans
                    fc_trans[iKxc] = f1d_temp.copy()
            fc_fft = fc_trans.transpose()

        else:
            nKy = self.shapeK_seq[0]
            fc_fft = np.empty([nKyc, nKxc], np.complex128)
            for iKyc in range(nKyc):
                if iKyc <= nKyc/2:
                    iKy = iKyc
                else:
                    kynodim = iKyc - nKyc
                    iKy = kynodim + nKy
                for iKxc in range(nKxc):
                    fc_fft[iKyc, iKxc] = f_fft[iKy, iKxc]
        return fc_fft

    def fft_loc_from_coarse_seq(self, fc_fft, shapeK_loc_coarse):
        """Return a large field in K space."""
        nKyc = shapeK_loc_coarse[0]
        nKxc = shapeK_loc_coarse[1]

        if nb_proc > 1:
            nKy = self.shapeK_seq[0]
            f_fft = self.create_arrayK(value=0.)
            fc_trans = fc_fft.transpose()

            for iKxc in range(nKxc):
                kx = self.deltakx*iKxc
                rank_iKx, iKxloc, iKyloc = self.where_is_wavenumber(kx, 0.)
                fc1D = fc_trans[iKxc]
                if rank_iKx != 0:
                    # message fc1D
                    fc1D = np.ascontiguousarray(fc1D)
                    if rank == 0:
                        comm.Send(fc1D, dest=rank_iKx, tag=iKxc)
                    elif rank == rank_iKx:
                        comm.Recv(fc1D, source=0, tag=iKxc)
                if rank == rank_iKx:
                    # copy
                    for iKyc in range(nKyc):
                        if iKyc <= nKyc/2:
                            iKy = iKyc
                        else:
                            kynodim = iKyc - nKyc
                            iKy = kynodim + nKy
                        f_fft[iKxloc, iKy] = fc1D[iKyc]

        else:
            nKy = self.shapeK_seq[0]
            nKx = self.shapeK_seq[1]
            f_fft = np.zeros([nKy, nKx], np.complex128)
            for iKyc in range(nKyc):
                if iKyc <= nKyc/2:
                    iKy = iKyc
                else:
                    kynodim = iKyc - nKyc
                    iKy = kynodim + nKy
                for iKxc in range(nKxc):
                    f_fft[iKy, iKxc] = fc_fft[iKyc, iKxc]
        return f_fft

    def compute_increments_dim1(self, var, irx):
        """Compute the increments of var over the dim 1."""
        return compute_increments_dim1(var, int(irx))

    def pdf_normalized(self, field, nb_bins=100):
        """Compute the normalized pdf"""

        field_max = field.max()
        field_min = field.min()
        # field_mean = field.mean()

        if nb_proc > 1:
            field_max = comm.allreduce(field_max, op=MPI.MAX)
            field_min = comm.allreduce(field_min, op=MPI.MIN)
            # field_mean = comm.allreduce(field_min, op=MPI.SUM)/nb_proc

        # rms = np.sqrt(np.mean( (field-field_mean)**2 ))
        # range_min = field_mean - 20*rms
        # range_max = field_mean + 20*rms

        # range_min = max(field_min, range_min)
        # range_max = min(field_max, range_max)

        range_min = field_min
        range_max = field_max

        if nb_proc == 1:
            pdf, bin_edges = np.histogram(field, bins=nb_bins,
                                          normed=True,
                                          range=(range_min, range_max))
        else:
            hist, bin_edges = np.histogram(field, bins=nb_bins,
                                           range=(range_min, range_max))
            hist = comm.allreduce(hist, op=MPI.SUM)
            pdf = hist/((bin_edges[1]-bin_edges[0])*hist.sum())
        return pdf, bin_edges

    def where_is_wavenumber(self, kx_approx, ky_approx):
        ikx_seq = int(np.round(kx_approx/self.deltakx))
        if ikx_seq >= self.nkx_seq:
            raise ValueError('not good :-) ikx_seq >= self.nkx_seq')

        if self.is_sequential:
            rank_k = 0
            ikx_loc = ikx_seq
        else:
            if self.SAME_SIZE_IN_ALL_PROC:
                rank_k = int(np.floor(float(ikx_seq)/self.nkx_loc))
                if ikx_seq >= self.nkx_loc * mpi.nb_proc:
                    raise ValueError(
                        'not good :-) ikx_seq >= self.nkx_loc * mpi.nb_proc\n'
                        'ikx_seq, self.nkx_loc, mpi.nb_proc = '
                        '{}, {}, {}'.format(
                            ikx_seq, self.nkx_loc, mpi.nb_proc))
            else:
                rank_k = 0
                while (rank_k < self.nb_proc-1 and
                       (not (self.iKxloc_start_rank[rank_k] <= ikx_seq and
                             ikx_seq < self.iKxloc_start_rank[rank_k+1]))):
                    rank_k += 1

            ikx_loc = ikx_seq - self.iKxloc_start_rank[rank_k]

        iky_loc = int(np.round(ky_approx/self.deltaky))
        if iky_loc < 0:
            iky_loc = self.nky_loc+iky_loc

        if self.is_transposed:
            ik0_loc = ikx_loc
            ik1_loc = iky_loc
        else:
            ik0_loc = iky_loc
            ik1_loc = ikx_loc

        return rank_k, ik0_loc, ik1_loc

    def uxuyfft_from_psifft(self, psi_fft):
        px_psi_fft, py_psi_fft = self.gradfft_from_fft(psi_fft)
        ux_fft = -py_psi_fft
        uy_fft = px_psi_fft
        return ux_fft, uy_fft

    def rotfft_from_psifft(self, psi_fft):
        rot_fft = -self.K2*psi_fft
        return rot_fft

    def uxuyetafft_from_qfft(self, q_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params
        K2 = self.K2
        K2_not0 = self.K2_not0
        rot_fft = K2*q_fft/(K2_not0+params.kd2)
        if rank == 0:
            rot_fft[0, 0] = 0.
        ux_fft, uy_fft = self.vecfft_from_rotfft(rot_fft)

        if params.f == 0:
            eta_fft = self.create_arrayK(value=0)
        else:
            eta_fft = -params.f*q_fft/(K2_not0+params.kd2)/params.c2
        if rank == 0:
            eta_fft[0, 0] = 0.
        return ux_fft, uy_fft, eta_fft

    def uxuyetafft_from_afft(self, a_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params
        # K2 = self.K2
        K2_not0 = self.K2_not0

        if params.f == 0:
            rot_fft = self.create_arrayK(value=0)
        else:
            rot_fft = params.f*a_fft/(K2_not0+params.kd2)
        if rank == 0:
            rot_fft[0, 0] = 0.
        ux_fft, uy_fft = self.vecfft_from_rotfft(rot_fft)

        eta_fft = a_fft/(K2_not0+params.kd2)
        if rank == 0:
            eta_fft[0, 0] = 0.
        return ux_fft, uy_fft, eta_fft

    def rotfft_from_qfft(self, q_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params
        K2 = self.K2
        K2_not0 = self.K2_not0
        rot_fft = K2*q_fft/(K2_not0+params.kd2)
        if rank == 0:
            rot_fft[0, 0] = 0.
        return rot_fft

    def rotfft_from_afft(self, a_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params
        # K2 = self.K2
        K2_not0 = self.K2_not0
        if params.f == 0:
            rot_fft = self.create_arrayK(value=0)
        else:
            rot_fft = params.f*a_fft/(K2_not0+params.kd2)
        if rank == 0:
            rot_fft[0, 0] = 0.
        return rot_fft

    def afft_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft,
                             params=None):
        if params is None:
            params = self.params
        rot_fft = self.rotfft_from_vecfft(ux_fft, uy_fft)
        a_fft = self.K2*eta_fft
        if params.f != 0:
            a_fft += params.f/params.c2*rot_fft
        return a_fft

    def etafft_from_qfft(self, q_fft, params=None):
        """Compute eta in Fourier space."""
        if params is None:
            params = self.params
        K2_not0 = self.K2_not0
        if params.f == 0:
            eta_fft = self.create_arrayK(value=0)
        else:
            eta_fft = -params.f/params.c2*q_fft/(K2_not0+params.kd2)
        if rank == 0:
            eta_fft[0, 0] = 0.
        return eta_fft

    def etafft_from_afft(self, a_fft, params=None):
        """Compute eta in Fourier space."""
        if params is None:
            params = self.params
        K2_not0 = self.K2_not0
        eta_fft = a_fft/(K2_not0+params.kd2)
        if rank == 0:
            eta_fft[0, 0] = 0.
        return eta_fft

    def etafft_from_aqfft(self, a_fft, q_fft, params=None):
        """Compute eta in Fourier space."""
        if params is None:
            params = self.params
        K2_not0 = self.K2_not0
        if params.f == 0:
            eta_fft = a_fft/K2_not0
        else:
            eta_fft = (a_fft - params.f/params.c2*q_fft)/(
                K2_not0+params.kd2)
        if rank == 0:
            eta_fft[0, 0] = 0.
        return eta_fft

    def qdafft_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft, params=None):
        if params is None:
            params = self.params
        div_fft = self.divfft_from_vecfft(ux_fft, uy_fft)
        rot_fft = self.rotfft_from_vecfft(ux_fft, uy_fft)
        q_fft = rot_fft - params.f*eta_fft
        ageo_fft = params.f/params.c2*rot_fft + self.K2*eta_fft
        return q_fft, div_fft, ageo_fft

    def apamfft_from_adfft(self, a_fft, d_fft):
        """Return the eigen modes ap and am."""
        Delta_a_fft = self.Kappa_over_ic * d_fft
        ap_fft = 0.5*(a_fft + Delta_a_fft)
        am_fft = 0.5*(a_fft - Delta_a_fft)
        return ap_fft, am_fft

    def divfft_from_apamfft(self, ap_fft, am_fft):
        """Return div from the eigen modes ap and am."""
        # cdef Py_ssize_t rank = self.rank

        Delta_a_fft = ap_fft - am_fft
        n0 = self.nK0_loc
        n1 = self.nK1_loc
        Kappa_over_ic = self.Kappa_over_ic
        d_fft = np.empty([n0, n1], dtype=np.complex128)

        for i0 in range(n0):
            for i1 in range(n1):
                if i0 == 0 and i1 == 0 and rank == 0:
                    d_fft[i0, i1] = 0.
                else:
                    d_fft[i0, i1] = (
                        Delta_a_fft[i0, i1]/Kappa_over_ic[i0, i1])
        return d_fft

    def qapamfft_from_etafft(self, eta_fft, params=None):
        """eta (fft) ---> q, ap, am (fft)"""
        if params is None:
            params = self.params

        q_fft = -params.f * eta_fft
        ap_fft = 0.5 * self.K2 * eta_fft
        am_fft = ap_fft.copy()
        return q_fft, ap_fft, am_fft

    def pxffft_from_fft(self, f_fft):
        """Return the gradient of f_fft in spectral space."""

        n0 = self.nK0_loc
        n1 = self.nK1_loc

        KX = self.KX

        px_f_fft = np.empty([n0, n1], dtype=np.complex128)

        if f_fft.dtype == np.float64:
            ff_fft = f_fft
            for i0 in range(n0):
                for i1 in range(n1):
                    px_f_fft[i0, i1] = 1j * KX[i0, i1]*ff_fft[i0, i1]
        else:
            fc_fft = f_fft
            for i0 in range(n0):
                for i1 in range(n1):
                    px_f_fft[i0, i1] = 1j * KX[i0, i1]*fc_fft[i0, i1]

        return px_f_fft

    def pyffft_from_fft(self, f_fft):
        """Return the gradient of f_fft in spectral space."""

        n0 = self.nK0_loc
        n1 = self.nK1_loc

        KY = self.KY

        py_f_fft = np.empty([n0, n1], dtype=np.complex128)

        if f_fft.dtype == np.float64:
            ff_fft = f_fft
            for i0 in range(n0):
                for i1 in range(n1):
                    py_f_fft[i0, i1] = 1j * KY[i0, i1]*ff_fft[i0, i1]
        else:
            fc_fft = f_fft
            for i0 in range(n0):
                for i1 in range(n1):
                    py_f_fft[i0, i1] = 1j * KY[i0, i1]*fc_fft[i0, i1]

        return py_f_fft

    def mean_space(self, field):

        mean_field = np.mean(field)
        if not self.is_sequential:
            mean_field = self.comm.allreduce(mean_field, op=MPI.SUM)
            mean_field /= nb_proc
        return mean_field

    def uxuyetafft_from_qapamfft(self, q_fft, ap_fft, am_fft):
        """q, ap, am (fft) ---> ux, uy, eta (fft)"""
        a_fft = ap_fft + am_fft
        if rank == 0:
            a_fft[0, 0] = 0.
        div_fft = self.divfft_from_apamfft(ap_fft, am_fft)
        (uxa_fft, uya_fft, etaa_fft
         ) = self.uxuyetafft_from_afft(a_fft)
        (uxq_fft, uyq_fft, etaq_fft
         ) = self.uxuyetafft_from_qfft(q_fft)
        uxd_fft, uyd_fft = self.vecfft_from_divfft(div_fft)
        ux_fft = uxa_fft + uxq_fft + uxd_fft
        uy_fft = uya_fft + uyq_fft + uyd_fft
        eta_fft = etaa_fft + etaq_fft
        if rank == 0:
            ux_fft[0, 0] = 0.5 * (ap_fft[0, 0] + am_fft[0, 0])
            uy_fft[0, 0] = 0.5j * (am_fft[0, 0] - ap_fft[0, 0])
        return ux_fft, uy_fft, eta_fft

    def laplacian2_fft(self, a_fft):
        return laplacian2_fft(a_fft, self.K4)

    def invlaplacian2_fft(self, a_fft):
        return invlaplacian2_fft(a_fft, self.K4_not0, rank)
