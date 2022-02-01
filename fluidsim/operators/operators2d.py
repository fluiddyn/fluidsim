"""Operators 2d (:mod:`fluidsim.operators.operators2d`)
=======================================================

Provides

.. autoclass:: OperatorsPseudoSpectral2D
   :members:
   :private-members:

"""

from warnings import warn
from random import uniform
import sys

import numpy as np

from transonic import boost, Array, Transonic
from fluiddyn.util import mpi
from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D as _Operators

from fluidsim.base.params import Parameters
from ..base.setofvariables import SetOfVariables
from .. import _is_testing
from .base import OperatorBase

ts = Transonic()

if (
    not ts.is_transpiling
    and not ts.is_compiled
    and not _is_testing
    and "sphinx" not in sys.modules
):
    warn(
        "operators2d.py has to be pythranized to be efficient! "
        "Install pythran and recompile."
    )
elif ts.is_transpiling:
    _Operators = object


Af = Array[np.float64, "2d"]
Ac = Array[np.complex128, "2d"]


@boost
def laplacian_fft(a_fft: Ac, Kn: Af):
    """Compute the n-th order Laplacian."""
    return a_fft * Kn


@boost
def invlaplacian_fft(a_fft: Ac, Kn_not0: Af, rank: int):
    """Compute the n-th order inverse Laplacian."""
    invlap_afft = a_fft / Kn_not0
    if rank == 0:
        invlap_afft[0, 0] = 0.0
    return invlap_afft


@boost
def compute_increments_dim1(var: Af, irx: int):
    """Compute the increments of var over the dim 1."""
    n1 = var.shape[1]
    n1new = n1 - irx
    # bug for gast 0.4.0 (https://github.com/serge-sans-paille/gast/issues/48)
    inc_var = var[:, irx:n1] - var[:, 0:n1new]
    return inc_var


if not ts.is_transpiling:
    nb_proc = mpi.nb_proc
    rank = mpi.rank
else:
    nb_proc = 1
    rank = 0

if nb_proc > 1:
    MPI = mpi.MPI
    comm = mpi.comm


@boost
class OperatorsPseudoSpectral2D(_Operators, OperatorBase):

    _has_to_dealiase: bool
    where_dealiased: "uint8[:, :]"
    KX: Af
    KY: Af
    deltax: float
    deltay: float

    @classmethod
    def _create_default_params(cls):
        params = Parameters(tag="params", attribs={"ONLY_COARSE_OPER": False})
        cls._complete_params_with_default(params)
        return params

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""
        attribs = {
            "type_fft": "default",
            "coef_dealiasing": 2.0 / 3,
            "nx": 48,
            "ny": 48,
            "Lx": 8,
            "Ly": 8,
            "truncation_shape": "cubic",
            "NO_SHEAR_MODES": False,
            "NO_KY0": False,
        }
        params._set_child("oper", attribs=attribs)

    def __init__(self, params):

        self.params = params
        self.axes = ("y", "x")
        nx = int(params.oper.nx)
        ny = int(params.oper.ny)

        if params.oper.nx != nx:
            raise ValueError(
                "params.oper.nx != int(params.oper.nx); "
                "({})".format(params.oper.nx)
            )

        if params.oper.ny != ny:
            raise ValueError(
                "params.oper.ny != int(params.oper.ny); "
                "({})".format(params.oper.ny)
            )

        params.oper.nx = nx
        params.oper.ny = ny

        if params.ONLY_COARSE_OPER:
            nx = ny = 4

        super().__init__(
            nx,
            ny,
            params.oper.Lx,
            params.oper.Ly,
            fft=params.oper.type_fft,
            coef_dealiasing=params.oper.coef_dealiasing,
        )

        # compatibility for fluidfft <= 0.3.0
        if not hasattr(self, "oper_fft") and hasattr(self, "opfft"):
            self.oper_fft = self.opfft

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
            self.iKxloc_start, _ = self.oper_fft.get_seq_indices_first_K()
            self.iKxloc_start_rank = np.array(comm.allgather(self.iKxloc_start))

            nkx_loc_rank = np.array(comm.allgather(self.nkx_loc))
            a = nkx_loc_rank
            self.SAME_SIZE_IN_ALL_PROC = (a >= a.max()).all()

        self._reinit_truncation()

        try:
            NO_SHEAR_MODES = self.params.oper.NO_SHEAR_MODES
        except AttributeError:
            pass
        else:
            if NO_SHEAR_MODES:
                COND_NOSHEAR = abs(self.KX) == 0.0
                self.where_dealiased = np.array(
                    np.logical_or(COND_NOSHEAR, self.where_dealiased),
                    dtype=np.uint8,
                )

        try:
            NO_KY0 = self.params.oper.NO_KY0
        except AttributeError:
            pass
        else:
            if NO_KY0:
                COND_NO_KY0 = abs(self.KY) == 0.0
                self.where_dealiased = np.array(
                    np.logical_or(COND_NO_KY0, self.where_dealiased),
                    dtype=np.uint8,
                )

    def get_region_multiple_aliases(self):
        aliases_x = abs(self.KX) >= 2 / 3 * self.deltakx * self.nx / 2
        aliases_y = abs(self.KY) >= 2 / 3 * self.deltaky * self.ny / 2
        return aliases_x & aliases_y

    def dealiasing(self, *args):
        if not self._has_to_dealiase:
            return

        for thing in args:
            if isinstance(thing, SetOfVariables):
                self.dealiasing_setofvar(thing)
            elif isinstance(thing, np.ndarray):
                self.dealiasing_variable(thing)

    @boost
    def dealiasing_setofvar(self, sov: "complex128[][][]"):
        """Dealiasing of a setofvar arrays."""
        if self._has_to_dealiase:
            nk, n0, n1 = sov.shape

            for i0 in range(n0):
                for i1 in range(n1):
                    if self.where_dealiased[i0, i1]:
                        for ik in range(nk):
                            sov[ik, i0, i1] = 0.0

    def project_fft_on_realX_seq(self, f_fft):
        """Project the given field in spectral space such as its
        inverse fft is a real field."""

        nky_seq = self.shapeK_seq[0]

        iky_ky0 = 0
        iky_kyM = nky_seq // 2
        ikx_kx0 = 0
        # ikx_kxM = self.nkx_seq-1
        ikx_kxM = self.shapeK_seq[1] - 1

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

            f_fft[ikyp, ikx_kx0] = (f_kp_kx0 + f_kn_kx0.conjugate()) / 2
            f_fft[ikyn, ikx_kx0] = (
                (f_kp_kx0 + f_kn_kx0.conjugate()) / 2
            ).conjugate()

            f_kp_kxM = f_fft[ikyp, ikx_kxM]
            f_kn_kxM = f_fft[ikyn, ikx_kxM]

            f_fft[ikyp, ikx_kxM] = (f_kp_kxM + f_kn_kxM.conjugate()) / 2
            f_fft[ikyn, ikx_kxM] = (
                (f_kp_kxM + f_kn_kxM.conjugate()) / 2
            ).conjugate()
        return f_fft

    def project_fft_on_realX_slow(self, f_fft):
        return self.fft(self.ifft(f_fft))

    def coarse_seq_from_fft_loc(self, f_fft, shapeK_loc_coarse):
        """Return a coarse field in K space."""
        nkyc = shapeK_loc_coarse[0]
        nkxc = shapeK_loc_coarse[1]

        if nb_proc > 1:
            if not self.is_transposed:
                raise NotImplementedError()

            fc_trans = np.empty([nkxc, nkyc], np.complex128)
            nky = self.shapeK_seq[1]
            f1d_temp = np.empty([nkyc], np.complex128)

            for ikxc in range(nkxc):
                kx = self.deltakx * ikxc
                rank_ikx, ikxloc, ikyloc = self.where_is_wavenumber(kx, 0.0)
                if rank == rank_ikx:
                    # create f1d_temp
                    for ikyc in range(nkyc):
                        if ikyc <= nkyc / 2:
                            iky = ikyc
                        else:
                            kynodim = ikyc - nkyc
                            iky = kynodim + nky
                        f1d_temp[ikyc] = f_fft[ikxloc, iky]

                if rank_ikx != 0:
                    # message f1d_temp
                    if rank == 0:
                        # print('f1d_temp', f1d_temp, f1d_temp.dtype)
                        comm.Recv(
                            [f1d_temp, MPI.DOUBLE_COMPLEX],
                            source=rank_ikx,
                            tag=ikxc,
                        )
                    elif rank == rank_ikx:
                        comm.Send(
                            [f1d_temp, MPI.DOUBLE_COMPLEX], dest=0, tag=ikxc
                        )
                if rank == 0:
                    # copy into fc_trans
                    fc_trans[ikxc] = f1d_temp.copy()
            fc_fft = fc_trans.transpose()

        else:
            nky = self.shapeK_seq[0]
            fc_fft = np.empty([nkyc, nkxc], np.complex128)
            for ikyc in range(nkyc):
                if ikyc <= nkyc / 2:
                    iky = ikyc
                else:
                    kynodim = ikyc - nkyc
                    iky = kynodim + nky
                for ikxc in range(nkxc):
                    fc_fft[ikyc, ikxc] = f_fft[iky, ikxc]

        # fc_fft[nkyc//2] *= 2

        # energy_coarse = self.sum_wavenumbers(abs(fc_fft)**2)
        # energy_global = self.sum_wavenumbers(abs(f_fft)**2)
        # print('energy_coarse = {}'.format(energy_coarse))
        # print('energy_global = {}'.format(energy_global))

        # assert energy_global == energy_coarse

        return fc_fft

    # def fft_loc_from_coarse_seq(self, fc_fft, shapeK_loc_coarse):
    #     """Return a large field in K space."""
    #     nkyc = shapeK_loc_coarse[0]
    #     nkxc = shapeK_loc_coarse[1]

    #     if nb_proc > 1:
    #         nky = self.shapeK_seq[0]
    #         f_fft = self.create_arrayK(value=0.)
    #         fc_trans = fc_fft.transpose()

    #         for ikxc in range(nkxc):
    #             kx = self.deltakx*ikxc
    #             rank_ikx, ikxloc, ikyloc = self.where_is_wavenumber(kx, 0.)
    #             fc1D = fc_trans[ikxc]
    #             if rank_ikx != 0:
    #                 # message fc1D
    #                 fc1D = np.ascontiguousarray(fc1D)
    #                 if rank == 0:
    #                     comm.Send(fc1D, dest=rank_ikx, tag=ikxc)
    #                 elif rank == rank_ikx:
    #                     comm.Recv(fc1D, source=0, tag=ikxc)
    #             if rank == rank_ikx:
    #                 # copy
    #                 for ikyc in range(nkyc):
    #                     if ikyc <= nkyc/2:
    #                         iky = ikyc
    #                     else:
    #                         kynodim = ikyc - nkyc
    #                         iky = kynodim + nky
    #                     f_fft[ikxloc, iky] = fc1D[ikyc]

    #     else:
    #         nky = self.shapeK_seq[0]
    #         nkx = self.shapeK_seq[1]
    #         f_fft = np.zeros([nky, nkx], np.complex128)
    #         for ikyc in range(nkyc):
    #             if ikyc <= nkyc/2:
    #                 iky = ikyc
    #             else:
    #                 kynodim = ikyc - nkyc
    #                 iky = kynodim + nky
    #             for ikxc in range(nkxc):
    #                 f_fft[iky, ikxc] = fc_fft[ikyc, ikxc]
    #     return f_fft

    def compute_increments_dim1(self, var, irx):
        """Compute the increments of var over the dim 1."""
        return compute_increments_dim1(var, int(irx))

    def pdf_normalized(self, field, nb_bins=100):
        """Compute the normalized pdf"""

        field_max = field.max()
        field_min = field.min()

        if nb_proc > 1:
            field_max = comm.allreduce(field_max, op=MPI.MAX)
            field_min = comm.allreduce(field_min, op=MPI.MIN)

        range_min = field_min
        range_max = field_max

        if nb_proc == 1:
            pdf, bin_edges = np.histogram(
                field, bins=nb_bins, density=True, range=(range_min, range_max)
            )
        else:
            hist, bin_edges = np.histogram(
                field, bins=nb_bins, range=(range_min, range_max)
            )
            # memory leak related to this line for CPython 3.7.1
            # hist = comm.allreduce(hist, op=MPI.SUM)
            # workaround for CPython 3.7.0 and 3.7.1
            tmp = np.empty_like(hist)
            comm.Allreduce(hist, tmp, op=MPI.SUM)
            hist = tmp
            #
            pdf = hist / ((bin_edges[1] - bin_edges[0]) * hist.sum())
        return pdf, bin_edges

    def where_is_wavenumber(self, kx_approx, ky_approx):
        ikx_seq = int(np.round(kx_approx / self.deltakx))
        if ikx_seq >= self.nkx_seq:
            raise ValueError("not good :-) ikx_seq >= self.nkx_seq")

        if self.is_sequential:
            rank_k = 0
            ikx_loc = ikx_seq
        else:
            if self.SAME_SIZE_IN_ALL_PROC:
                rank_k = int(np.floor(float(ikx_seq) / self.nkx_loc))
                if ikx_seq >= self.nkx_loc * mpi.nb_proc:
                    raise ValueError(
                        "not good :-) ikx_seq >= self.nkx_loc * mpi.nb_proc\n"
                        "ikx_seq, self.nkx_loc, mpi.nb_proc = "
                        "{}, {}, {}".format(ikx_seq, self.nkx_loc, mpi.nb_proc)
                    )

            else:
                rank_k = 0
                while rank_k < self.nb_proc - 1 and (
                    not (
                        self.iKxloc_start_rank[rank_k] <= ikx_seq
                        and ikx_seq < self.iKxloc_start_rank[rank_k + 1]
                    )
                ):
                    rank_k += 1

            ikx_loc = ikx_seq - self.iKxloc_start_rank[rank_k]

        iky_loc = int(np.round(ky_approx / self.deltaky))
        if iky_loc < 0:
            iky_loc = self.nky_loc + iky_loc

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
        return self.laplacian_fft(psi_fft)

    def pxffft_from_fft(self, f_fft):
        """Return the gradient of f_fft in spectral space."""
        return 1j * self.KX * f_fft

    def pyffft_from_fft(self, f_fft):
        """Return the gradient of f_fft in spectral space."""
        return 1j * self.KY * f_fft

    def laplacian2_fft(self, a_fft):
        warn("Use oper.laplacian_fft instead.", PendingDeprecationWarning)
        return laplacian_fft(a_fft, self.K4)

    def invlaplacian2_fft(self, a_fft):
        warn("Use oper.invlaplacian_fft instead.", PendingDeprecationWarning)
        return invlaplacian_fft(a_fft, self.K4_not0, rank)

    def laplacian_fft(self, a_fft, order=2, negative=False):
        r"""Compute the n-th order Laplacian, :math:`\nabla^{n} \hat{a}`

        Parameters
        ----------
        a_fft : ndarray

        order: int, {2, 4, 8}, optional
            Order of the Laplacian operator

        negative: bool, optional
            Negative of the result.

        """
        sign = 1j**order
        if sign.imag != 0:
            raise ValueError(f"Order={order} should be even!")

        if negative:
            sign *= -1
        Kn = getattr(self, "K{}".format(int(order)))

        # Avoid unnecessary multiplication by unity
        if sign == 1:
            return laplacian_fft(a_fft, Kn)
        else:
            return sign * laplacian_fft(a_fft, Kn)

    def invlaplacian_fft(self, a_fft, order=2, negative=False):
        r"""Compute the n-th order inverse Laplacian, :math:`\nabla^{-n} \hat{a}`

        Parameters
        ----------
        a_fft : ndarray

        order: int, {2, 4, 8}, optional
            Order of the inverse Laplacian operator.

        negative: bool, optional
            Negative of the result.

        """
        sign = 1.0 / 1j**order
        if sign.imag != 0:
            raise ValueError(f"Order={order} should be even!")

        if negative:
            sign *= -1
        Kn_not0 = getattr(self, "K{}_not0".format(int(order)))

        # Avoid unnecessary multiplication by unity
        if sign == 1:
            return invlaplacian_fft(a_fft, Kn_not0, rank)
        else:
            return sign * invlaplacian_fft(a_fft, Kn_not0, rank)

    def put_coarse_array_in_array_fft(
        self, arr_coarse, arr, oper_coarse, shapeK_loc_coarse
    ):
        """Put the values contained in a coarse array in an array.

        Both arrays are in Fourier space.

        """
        if arr.ndim == 3:
            if mpi.rank == 0:
                if arr_coarse.ndim != 3:
                    raise ValueError

            for ikey in range(arr.shape[0]):
                if mpi.rank == 0:
                    arr2d_coarse = arr_coarse[ikey]
                else:
                    arr2d_coarse = None
                self.put_coarse_array_in_array_fft(
                    arr2d_coarse, arr[ikey], oper_coarse, shapeK_loc_coarse
                )
            return

        nKyc, nKxc = shapeK_loc_coarse

        if mpi.nb_proc > 1 and not self.is_sequential:
            if not self.is_transposed:
                raise NotImplementedError()

            nKy = self.shapeK_seq[1]

            if mpi.rank == 0:
                fck_fft = arr_coarse.transpose()

            for ikxc in range(nKxc):
                kx = self.deltakx * ikxc
                rank_ikx, ikxloc, ikyloc = self.where_is_wavenumber(kx, 0.0)

                if mpi.rank == 0:
                    fc1D = fck_fft[ikxc]

                if rank_ikx != 0:
                    # message fc1D
                    if mpi.rank == rank_ikx:
                        fc1D = np.empty([nKyc], dtype=np.complex128)
                    if mpi.rank == 0 or mpi.rank == rank_ikx:
                        fc1D = np.ascontiguousarray(fc1D)
                    if mpi.rank == 0:
                        mpi.comm.Send(
                            [fc1D, mpi.MPI.COMPLEX], dest=rank_ikx, tag=ikxc
                        )
                    elif mpi.rank == rank_ikx:
                        mpi.comm.Recv([fc1D, mpi.MPI.COMPLEX], source=0, tag=ikxc)
                if mpi.rank == rank_ikx:
                    # copy
                    for ikyc in range(nKyc):
                        if ikyc <= nKyc / 2.0:
                            iky = ikyc
                        else:
                            kynodim = ikyc - nKyc
                            iky = kynodim + nKy
                        arr[ikxloc, iky] = fc1D[ikyc]
        else:
            nKy = self.shapeK_seq[0]

            if not np.allclose(0.0, max(abs(arr_coarse[nKyc // 2, :]))):
                raise ValueError("any(arr_coarse[nKyc//2] != 0)")

            if not np.allclose(0.0, max(abs(arr_coarse[:, nKxc - 1]))):
                raise ValueError("any(arr_coarse[:, nKxc-1] != 0)")

            for ikyc in range(nKyc):
                if ikyc <= nKyc / 2.0:
                    iky = ikyc
                else:
                    kynodim = ikyc - nKyc
                    iky = kynodim + nKy

                for ikxc in range(nKxc):
                    arr[iky, ikxc] = arr_coarse[ikyc, ikxc]

    def get_grid1d_seq(self, axe="x"):

        if axe not in ("x", "y"):
            raise ValueError

        if self.params.ONLY_COARSE_OPER:
            number_points = getattr(self.params.oper, "n" + axe)
            length = getattr(self, "L" + axe)
            return np.linspace(0, length, number_points)
        else:
            return getattr(self, axe + "_seq")

    @boost
    def get_phases_random(self):
        # Not supported by Pythran 0.9.5!
        # alpha_x, alpha_y = np.random.uniform(-0.5, 0.5, 2)
        alpha_x, alpha_y = tuple(uniform(-0.5, 0.5) for _ in range(2))
        beta_x = alpha_x + 0.5 if alpha_x < 0 else alpha_x - 0.5
        beta_y = alpha_y + 0.5 if alpha_y < 0 else alpha_y - 0.5

        phase_alpha = (
            alpha_x * self.deltax * self.KX + alpha_y * self.deltay * self.KY
        )
        phase_beta = (
            beta_x * self.deltax * self.KX + beta_y * self.deltay * self.KY
        )
        return phase_alpha, phase_beta


# energy_arr = self.sum_wavenumbers(abs(arr)**2)
# energy_array_coarse_after = oper_coarse.sum_wavenumbers(
#     abs(arr_coarse)**2)
# print('energy_array_coarse_after  = ', energy_array_coarse_after)
# print('energy_arr                 = ', energy_arr)
