
import numpy as np

from fluiddyn.util import mpi

from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D as _Operators

from . import util2d_pythran
from .util2d_pythran import dealiasing_setofvar
from ..base.setofvariables import SetOfVariables

if not hasattr(util2d_pythran, '__pythran__'):
    raise ValueError('util2d_pythran has to be pythranized to be efficient! '
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
            type_fft = 'fft2d.with_fftw2d'

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
            if nb_proc > 1:
                self.project_fft_on_realX = self.project_fft_on_realX_slow
            else:
                self.project_fft_on_realX = self.project_fft_on_realX_seq

        if not self.is_sequential:

            self.iKxloc_start, _ = self.opfft.get_seq_indices_first_K()
            self.iKxloc_start_rank = np.array(
                comm.allgather(self.iKxloc_start))

            nkx_loc_rank = np.array(comm.allgather(self.nky_loc))
            a = nkx_loc_rank
            self.SAME_SIZE_IN_ALL_PROC = (a >= a.max()).all()

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

    def constant_arrayK(self, value=None, dtype=complex, shape='loc'):
        """Return a constant array in spectral space."""
        shape = shape.lower()
        if shape == 'loc':
            shapeK = self.shapeK_loc
        elif shape == 'seq':
            shapeK = self.shapeK_seq
        elif shape == 'gat':
            shapeK = self.shapeK_gat
        else:
            raise ValueError('shape should be "loc" or "seq"')
        if value is None:
            field_lm = np.empty(shapeK, dtype=dtype)
        elif value == 0:
            field_lm = np.zeros(shapeK, dtype=dtype)
        else:
            field_lm = value*np.ones(shapeK, dtype=dtype)
        return field_lm

    def constant_arrayX(self, value=None, dtype=np.float64, shape='loc'):
        """Return a constant array in real space."""
        shape = shape.lower()
        if shape == 'loc':
            shapeX = self.shapeX_loc
        elif shape == 'seq':
            shapeX = self.shapeX_seq
        else:
            raise ValueError('shape should be "loc" of "seq"')
        if value is None:
            field = np.empty(shapeX, dtype=dtype)
        elif value == 0:
            field = np.zeros(shapeX, dtype=dtype)
        else:
            field = value*np.ones(shapeX, dtype=dtype)
        return field

    def random_arrayK(self, shape='loc'):
        """Return a random array in spectral space."""
        shape = shape.lower()
        if shape == 'loc':
            shapeK = self.shapeK_loc
        elif shape == 'seq':
            shapeK = self.shapeK_seq
        elif shape == 'gat':
            shapeK = self.shapeK_gat
        else:
            raise ValueError('shape should be "loc", "gat" or "seq"')
        a_fft = (np.random.random(shapeK) +
                 1j*np.random.random(shapeK) - 0.5 - 0.5j)
        return a_fft

    def random_arrayX(self, shape='loc'):
        """Return a random array in real space."""
        shape = shape.lower()
        if shape == 'loc':
            shapeX = self.shapeX_loc
        elif shape == 'seq':
            shapeX = self.shapeX_seq
        else:
            raise ValueError('shape should be "loc" or "seq"')
        return np.random.random(shapeX)

    def project_fft_on_realX_seq(self, f_fft):
        """Project the given field in spectral space such as its
        inverse fft is a real field."""

        nky_seq = self.shapeK_seq[0]

        iky_ky0 = 0
        iky_kyM = nky_seq/2
        ikx_kx0 = 0
        # ikx_kxM = self.nkx_seq-1
        ikx_kxM = self.shapeK_seq[1]-1

        # first, some values have to be real
        f_fft[iky_ky0, ikx_kx0] = f_fft[iky_ky0, ikx_kx0].real
        f_fft[iky_ky0, ikx_kxM] = f_fft[iky_ky0, ikx_kxM].real
        f_fft[iky_kyM, ikx_kx0] = f_fft[iky_kyM, ikx_kx0].real
        f_fft[iky_kyM, ikx_kxM] = f_fft[iky_kyM, ikx_kxM].real

        # second, there are relations between some values
        for ikyp in xrange(1, iky_kyM):
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

    def project_fft_on_realX_slow(self, f_fft):
        return self.fft(self.ifft(f_fft))

    def coarse_seq_from_fft_loc(self, f_fft, shapeK_loc_coarse):
        """Return a coarse field in K space."""
        nKyc = shapeK_loc_coarse[0]
        nKxc = shapeK_loc_coarse[1]

        if nb_proc > 1:
            fc_trans = np.empty([nKxc, nKyc], np.complex128)
            nKy = self.shapeK_seq[0]
            f1D_temp = np.empty([nKyc], np.complex128)

            for iKxc in xrange(nKxc):
                kx = self.deltakx*iKxc
                rank_iKx, iKxloc, iKyloc = self.where_is_wavenumber(kx, 0.)
                if rank == rank_iKx:
                    # create f1D_temp
                    for iKyc in xrange(nKyc):
                        if iKyc <= nKyc/2:
                            iKy = iKyc
                        else:
                            kynodim = iKyc - nKyc
                            iKy = kynodim + nKy
                        f1D_temp[iKyc] = f_fft[iKxloc, iKy]

                if rank_iKx != 0:
                    # message f1D_temp
                    if rank == 0:
                        # print('f1D_temp', f1D_temp, f1D_temp.dtype)
                        comm.Recv(
                            [f1D_temp, MPI.DOUBLE_COMPLEX],
                            source=rank_iKx, tag=iKxc)
                    elif rank == rank_iKx:
                        comm.Send(
                            [f1D_temp, MPI.DOUBLE_COMPLEX],
                            dest=0, tag=iKxc)
                if rank == 0:
                    # copy into fc_trans
                    fc_trans[iKxc] = f1D_temp.copy()
            fc_fft = fc_trans.transpose()

        else:
            nKy = self.shapeK_seq[0]
            # nKx = self.shapeK_seq[1]
            fc_fft = np.empty([nKyc, nKxc], np.complex128)
            for iKyc in xrange(nKyc):
                if iKyc <= nKyc/2:
                    iKy = iKyc
                else:
                    kynodim = iKyc - nKyc
                    iKy = kynodim + nKy
                for iKxc in xrange(nKxc):
                    fc_fft[iKyc, iKxc] = f_fft[iKy, iKxc]
        return fc_fft

    def fft_loc_from_coarse_seq(self, fc_fft, shapeK_loc_coarse):
        """Return a large field in K space."""
        nKyc = shapeK_loc_coarse[0]
        nKxc = shapeK_loc_coarse[1]

        if nb_proc > 1:
            nKy = self.shapeK_seq[0]
            f_fft = self.constant_arrayK(value=0.)
            fc_trans = fc_fft.transpose()

            for iKxc in xrange(nKxc):
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
                    for iKyc in xrange(nKyc):
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
            for iKyc in xrange(nKyc):
                if iKyc <= nKyc/2:
                    iKy = iKyc
                else:
                    kynodim = iKyc - nKyc
                    iKy = kynodim + nKy
                for iKxc in xrange(nKxc):
                    f_fft[iKy, iKxc] = fc_fft[iKyc, iKxc]
        return f_fft

    def compute_increments_dim1(self, var, irx):
        """Compute the increments of var over the dim 1."""

        n0 = var.shape[0]
        n1 = var.shape[1]
        n1new = n1 - irx
        inc_var = np.empty([n0, n1new])
        for i0 in xrange(n0):
            for i1 in xrange(n1new):
                inc_var[i0, i1] = (var[i0, i1+irx] - var[i0, i1])
        return inc_var

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
