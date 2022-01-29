"""Operators 3d (:mod:`fluidsim.operators.operators3d`)
=======================================================

Provides

.. autoclass:: OperatorsPseudoSpectral3D
   :members:
   :private-members:

"""

from math import pi
from copy import deepcopy
from random import uniform

import numpy as np

from transonic import boost, Array, Transonic
from fluiddyn.util import mpi
from fluiddyn.util.mpi import nb_proc, rank
from fluidfft.fft3d.operators import OperatorsPseudoSpectral3D as _Operators

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.base.params import Parameters

from .operators2d import OperatorsPseudoSpectral2D as OpPseudoSpectral2D
from .. import _is_testing
from .base import OperatorBase

ts = Transonic()

Asov = Array[np.complex128, "4d"]
Aui8 = Array[np.uint8, "3d"]
Ac = Array[np.complex128, "3d"]
Af = Array[np.float64, "3d"]


@boost
def dealiasing_setofvar(sov: Asov, where_dealiased: Aui8):
    """Dealiasing 3d setofvar object.

    Parameters
    ----------

    sov : 4d ndarray
        A set of variables array.

    where_dealiased : 3d ndarray
        A 3d array of "booleans" (actually uint8).

    """
    nk, n0, n1, n2 = sov.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                if where_dealiased[i0, i1, i2]:
                    for ik in range(nk):
                        sov[ik, i0, i1, i2] = 0.0


@boost
def dealiasing_variable(ff_fft: Ac, where_dealiased: Aui8):
    """Dealiasing 3d array"""
    n0, n1, n2 = ff_fft.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                if where_dealiased[i0, i1, i2]:
                    ff_fft[i0, i1, i2] = 0.0


def dealiasing_setofvar_numpy(sov: Asov, where_dealiased: Aui8):
    for i in range(sov.shape[0]):
        sov[i][np.nonzero(where_dealiased)] = 0.0


def dealiasing_variable_numpy(ff_fft: Ac, where_dealiased: Aui8):
    ff_fft[np.nonzero(where_dealiased)] = 0.0


@boost
def compute_energy_from_1field(arr: Ac):
    return 0.5 * np.abs(arr) ** 2


@boost
def compute_energy_from_1field_with_coef(arr: Ac, coef: float):
    return 0.5 * coef * np.abs(arr) ** 2


@boost
def compute_energy_from_2fields(vx: Ac, vy: Ac):
    return 0.5 * (np.abs(vx) ** 2 + np.abs(vy) ** 2)


@boost
def compute_energy_from_3fields(vx: Ac, vy: Ac, vz: Ac):
    return 0.5 * (np.abs(vx) ** 2 + np.abs(vy) ** 2 + np.abs(vz) ** 2)


if not ts.is_transpiling and not ts.is_compiled and not _is_testing:
    # for example if Pythran is not available
    dealiasing_variable = dealiasing_variable_numpy
    dealiasing_setofvar = dealiasing_setofvar_numpy
elif ts.is_transpiling:
    _Operators = object


if nb_proc > 1:
    MPI = mpi.MPI
    comm = mpi.comm


@boost
class OperatorsPseudoSpectral3D(_Operators, OperatorBase):
    """Provides fast Fourier transform functions and 3D operators.

    Uses fft operators that implement the methods:

    - ifft
    - fft
    - get_shapeX_loc
    - get_shapeX_seq
    - get_shapeK_loc
    - get_shapeK_seq
    - get_dimX_K
    - get_seq_indices_first_K

    - get_k_adim_loc
    - sum_wavenumbers
    - build_invariant_arrayK_from_2d_indices12X

    """

    Kx: Af
    Ky: Af
    Kz: Af
    inv_K_square_nozero: Af
    inv_Kh_square_nozero: Af
    deltax: float
    deltay: float
    deltaz: float

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
            "type_fft2d": "sequential",
            "coef_dealiasing": 2.0 / 3,
            "nx": 48,
            "ny": 48,
            "nz": 48,
            "Lx": 2 * pi,
            "Ly": 2 * pi,
            "Lz": 2 * pi,
            "truncation_shape": "cubic",
            "NO_SHEAR_MODES": False,
        }
        params._set_child("oper", attribs=attribs)
        params.oper._set_doc(
            """

See the `documentation of fluidfft <http://fluidfft.readthedocs.io>`_ (in
particular of the `3d operator class
<http://fluidfft.readthedocs.io/en/latest/generated/fluidfft.fft3d.operators.html>`_).

type_fft: str

    Method for the FFT (as defined by fluidfft).

type_fft2d: str

    Method for the 2d FFT.

coef_dealiasing: float

    dealiasing coefficient.

nx: int

    Number of points over the x-axis (last dimension in the physical space).

ny: int

    Number of points over the y-axis (second dimension in the physical space).

nz: int

    Number of points over the z-axis (first dimension in the physical space).

Lx, Ly and Lz: float

    Length of the edges of the numerical domain.

"""
        )

    def __init__(self, params=None):

        self.params = params
        self.axes = ("z", "y", "x")

        params.oper.nx = int(params.oper.nx)
        params.oper.ny = int(params.oper.ny)
        params.oper.nz = int(params.oper.nz)

        if params.ONLY_COARSE_OPER:
            nx = ny = nz = 4
        else:
            nx = params.oper.nx
            ny = params.oper.ny
            nz = params.oper.nz

        super().__init__(
            nx,
            ny,
            nz,
            params.oper.Lx,
            params.oper.Ly,
            params.oper.Lz,
            fft=params.oper.type_fft,
            coef_dealiasing=params.oper.coef_dealiasing,
        )

        # compatibility for fluidfft <= 0.3.0
        if not hasattr(self, "oper_fft") and hasattr(self, "_op_fft"):
            self.oper_fft = self._op_fft

        # problem here type_fft
        params2d = deepcopy(params)
        params2d.oper.type_fft = params2d.oper.type_fft2d
        fft = params2d.oper.type_fft

        if (
            any([fft.startswith(s) for s in ["fluidfft.fft2d.", "fft2d."]])
            or fft in ("default", "sequential")
            or fft is None
        ):
            self.oper2d = OpPseudoSpectral2D(params2d)
        else:
            raise ValueError

        self.ifft2 = self.ifft2d = self.oper2d.ifft2
        self.fft2 = self.fft2d = self.oper2d.fft2
        if not self.is_sequential:
            self.iK0loc_start = self.seq_indices_first_K[0]
            self.nk0_loc, self.nk1_loc, self.nk2_loc = self.shapeK_loc

            self.iK0loc_start_rank = np.array(comm.allgather(self.iK0loc_start))
            nk2_loc = self.shapeK_loc[2]
            nk2_loc_rank = np.array(comm.allgather(nk2_loc))
            a = nk2_loc_rank
            self.SAME_SIZE_IN_ALL_PROC = (a >= a.max()).all()
            self.dimX_K = self.oper_fft.get_dimX_K()
        else:
            self.SAME_SIZE_IN_ALL_PROC = True

        self._reinit_truncation()

        try:
            NO_SHEAR_MODES = self.params.oper.NO_SHEAR_MODES
        except AttributeError:
            pass
        else:
            if NO_SHEAR_MODES:
                COND_NOSHEAR = self.Kx**2 + self.Ky**2 == 0.0
                self.where_dealiased = np.array(
                    np.logical_or(COND_NOSHEAR, self.where_dealiased),
                    dtype=np.uint8,
                )

    def get_region_multiple_aliases(self):
        aliases_x = abs(self.Kx) >= 2 / 3 * self.deltakx * self.nx / 2
        aliases_y = abs(self.Ky) >= 2 / 3 * self.deltaky * self.ny / 2
        aliases_z = abs(self.Kz) >= 2 / 3 * self.deltakz * self.nz / 2
        return (
            (aliases_x & aliases_y)
            | (aliases_y & aliases_z)
            | (aliases_z & aliases_x)
        )

    @property
    def K2_not0(self):
        K2_not0 = np.copy(self.K2)
        if sum(self.seq_indices_first_K) == 0:
            K2_not0[0, 0, 0] = 1e-14
        return K2_not0

    @property
    def K4(self):
        return self.K2**2

    def build_invariant_arrayX_from_2d_indices12X(self, arr2d, oper2d=None):
        """Build a 3D array from a 2D array"""
        if oper2d is None:
            oper2d = self.oper2d

        if mpi.nb_proc == 1:
            return self.oper_fft.build_invariant_arrayX_from_2d_indices12X(
                oper2d.oper_fft, arr2d
            )

        if mpi.rank > 0:
            assert arr2d is None
        arr2d = mpi.comm.bcast(arr2d, root=0)

        n0_loc, n1_loc, n2_loc = self.shapeX_loc

        if mpi.rank == 0:
            shapeX_loc_2d = oper2d.shapeX_loc
        else:
            shapeX_loc_2d = None
        shapeX_loc_2d = mpi.comm.bcast(shapeX_loc_2d, root=0)

        if shapeX_loc_2d != (n1_loc, n2_loc):
            raise NotImplementedError
            # ind0seq_first, ind1seq_first, ind2seq_first = \
            #     self.oper_fft.get_seq_indices_first_K()

        return np.stack(n0_loc * [arr2d])

    def build_invariant_arrayK_from_2d_indices12X(self, arr2d):
        """Build a 3D array from a 2D array"""
        return self.oper_fft.build_invariant_arrayK_from_2d_indices12X(
            self.oper2d, arr2d
        )

    def dealiasing(self, *args):
        """Dealiasing of SetOfVariables or np.ndarray"""
        for thing in args:
            if isinstance(thing, SetOfVariables):
                dealiasing_setofvar(thing, self.where_dealiased)
            elif isinstance(thing, np.ndarray):
                dealiasing_variable(thing, self.where_dealiased)

    def put_coarse_array_in_array_fft(
        self, arr_coarse, arr, oper_coarse, shapeK_loc_coarse
    ):
        """Put the values contained in a coarse array in an array.

        Both arrays are in Fourier space.

        """
        if arr.ndim == 4:
            if rank == 0:
                if arr_coarse.ndim != 4:
                    raise ValueError

            for ikey in range(arr.shape[0]):
                if rank == 0:
                    arr3d_coarse = arr_coarse[ikey]
                else:
                    arr3d_coarse = None
                self.put_coarse_array_in_array_fft(
                    arr3d_coarse, arr[ikey], oper_coarse, shapeK_loc_coarse
                )
            return

        nkzc, nkyc, nkxc = shapeK_loc_coarse

        if nb_proc > 1:
            nk0, nk1, nk2 = self.shapeK_seq
            if mpi.rank == 0:
                if self.dimX_K == (1, 0, 2):
                    fck_fft = np.zeros((nkyc, nkzc, nkxc), dtype=np.complex128)
                    for i0 in range(nkzc):
                        for i1 in range(nkyc):
                            fck_fft[i1, i0, :] = arr_coarse[i0, i1, :]
                if self.dimX_K == (2, 1, 0):
                    fck_fft = np.zeros((nkxc, nkyc, nkzc), dtype=np.complex128)
                    for i0 in range(nkzc):
                        for i1 in range(nkyc):
                            for i2 in range(nkxc):
                                fck_fft[i2, i1, i0] = arr_coarse[i0, i1, i2]

            if self.dimX_K == (1, 0, 2):
                nk1c, nk0c, nk2c = shapeK_loc_coarse
            elif self.dimX_K == (2, 1, 0):
                nk2c, nk1c, nk0c = shapeK_loc_coarse

            for ik0c in range(nk0c):
                ik1c = 0
                ik2c = 0
                ik0 = _ik_from_ikc(ik0c, nk0c, nk0)
                rank_ik, ik0loc, ik1loc, ik2loc = self.where_is_wavenumber(
                    ik0, ik1c, ik2c
                )
                if mpi.rank == 0:
                    fc1D = fck_fft[ik0c, :, :]
                if rank_ik != 0:
                    # message fc1D
                    if mpi.rank == rank_ik:
                        fc1D = np.empty([nk1c, nk2c], dtype=np.complex128)
                    if mpi.rank == 0 or mpi.rank == rank_ik:
                        fc1D = np.ascontiguousarray(fc1D)
                    if mpi.rank == 0:
                        mpi.comm.Send(
                            [fc1D, mpi.MPI.COMPLEX], dest=rank_ik, tag=ik0c
                        )
                    elif mpi.rank == rank_ik:
                        mpi.comm.Recv([fc1D, mpi.MPI.COMPLEX], source=0, tag=ik0c)
                if mpi.rank == rank_ik:
                    # copy
                    for ik1c in range(nk1c):
                        ik1 = _ik_from_ikc(ik1c, nk1c, nk1)
                        arr[ik0loc, ik1, 0:nk2c] = fc1D[ik1c, :]

        else:
            nkz, nky, nkx = self.shapeK_seq
            for ikzc in range(nkzc):
                ikz = _ik_from_ikc(ikzc, nkzc, nkz)
                for ikyc in range(nkyc):
                    iky = _ik_from_ikc(ikyc, nkyc, nky)
                    for ikxc in range(nkxc):
                        arr[ikz, iky, ikxc] = arr_coarse[ikzc, ikyc, ikxc]

    def coarse_seq_from_fft_loc(self, f_fft, shapeK_loc_coarse):
        """Return a coarse field in K space."""
        nkzc, nkyc, nkxc = shapeK_loc_coarse
        if nb_proc > 1:
            if self.shapeK_seq[1:2] != self.shapeK_loc[1:2]:
                raise NotImplementedError()

            if self.dimX_K == (1, 0, 2):
                nk1c, nk0c, nk2c = shapeK_loc_coarse
            elif self.dimX_K == (2, 1, 0):
                nk2c, nk1c, nk0c = shapeK_loc_coarse

            fc_fft_tmp = np.empty([nk0c, nk1c, nk2c], np.complex128)
            nk0, nk1, nk2 = self.shapeK_seq
            f1d_temp = np.empty([nk1c, nk2c], np.complex128)

            for ik0c in range(nk0c):
                ik1c = 0
                ik2c = 0
                ik0 = _ik_from_ikc(ik0c, nk0c, nk0)
                rank_ik, ik0loc, ik1loc, ik1loc = self.where_is_wavenumber(
                    ik0, ik1c, ik2c
                )
                if rank == rank_ik:
                    # create f1d_temp
                    for ik1c in range(nk1c):
                        ik1 = _ik_from_ikc(ik1c, nk1c, nk1)
                        f1d_temp[ik1c, :] = f_fft[ik0loc, ik1, 0:nk2c]

                if rank_ik != 0:
                    # message f1d_temp
                    if rank == 0:
                        comm.Recv(
                            [f1d_temp, MPI.DOUBLE_COMPLEX],
                            source=rank_ik,
                            tag=ik0c,
                        )
                    elif rank == rank_ik:
                        comm.Send(
                            [f1d_temp, MPI.DOUBLE_COMPLEX], dest=0, tag=ik0c
                        )
                if rank == 0:
                    # copy into fc_fft
                    fc_fft_tmp[ik0c] = f1d_temp.copy()
            fc_fft = np.zeros(shapeK_loc_coarse, dtype=np.complex128)
            if rank == 0:
                if self.dimX_K == (1, 0, 2):
                    for i0 in range(nkzc):
                        for i1 in range(nkyc):
                            fc_fft[i0, i1, :] = fc_fft_tmp[i1, i0, :]
                if self.dimX_K == (2, 1, 0):
                    for i0 in range(nkzc):
                        for i1 in range(nkyc):
                            for i2 in range(nkxc):
                                fc_fft[i0, i1, i2] = fc_fft_tmp[i2, i1, i0]
        else:
            fc_fft = np.empty(shapeK_loc_coarse, np.complex128)
            nkz, nky, nkx = self.shapeK_seq
            for ikzc in range(nkzc):
                ikz = _ik_from_ikc(ikzc, nkzc, nkz)
                for ikyc in range(nkyc):
                    iky = _ik_from_ikc(ikyc, nkyc, nky)
                    for ikxc in range(nkxc):
                        fc_fft[ikzc, ikyc, ikxc] = f_fft[ikz, iky, ikxc]
        return fc_fft

    def where_is_wavenumber(self, ik0, ik1, ik2):
        """Give local indices and rank from the sequential indices"""
        nk0_seq, nk1_seq, nk2_seq = self.shapeK_seq

        if ik0 >= nk0_seq:
            raise ValueError("not good :-) ik0_seq >= nk0_seq")

        if nb_proc == 1:
            rank_k = 0
            ik0_loc = ik0
        else:
            if self.SAME_SIZE_IN_ALL_PROC:
                rank_k = int(np.floor(float(ik0) / self.nk0_loc))
                if ik0 >= self.nk0_loc * nb_proc:
                    raise ValueError(
                        "not good :-) ik0_seq >= self.nk0_loc * mpi.nb_proc\n"
                        "ik0_seq, self.nk0_loc, mpi.nb_proc = "
                        f"{ik0}, {self.nk0_loc}, {nb_proc}"
                    )

            else:
                rank_k = 0
                while rank_k < self.nb_proc - 1 and (
                    not (
                        self.iK0loc_start_rank[rank_k] <= ik0
                        and ik0 < self.iK0loc_start_rank[rank_k + 1]
                    )
                ):
                    rank_k += 1

            ik0_loc = ik0 - self.iK0loc_start_rank[rank_k]

        ik1_loc = ik1
        if ik1_loc < 0:
            ik1_loc = self.nk1_loc + ik1_loc

        ik2_loc = ik2
        if ik2_loc < 0:
            ik2_loc = self.nk2_loc + ik2_loc

        return rank_k, ik0_loc, ik1_loc, ik2_loc

    @boost
    def urudfft_from_vxvyfft(self, vx_fft: Ac, vy_fft: Ac):
        """Compute toroidal and poloidal horizontal velocities.

        urx_fft, ury_fft contain shear modes!

        """
        inv_Kh_square_nozero = self.Kx**2 + self.Ky**2
        inv_Kh_square_nozero[inv_Kh_square_nozero == 0] = 1e-14
        inv_Kh_square_nozero = 1 / inv_Kh_square_nozero

        kdotu_fft = self.Kx * vx_fft + self.Ky * vy_fft
        udx_fft = kdotu_fft * self.Kx * inv_Kh_square_nozero
        udy_fft = kdotu_fft * self.Ky * inv_Kh_square_nozero

        urx_fft = vx_fft - udx_fft
        ury_fft = vy_fft - udy_fft

        return urx_fft, ury_fft, udx_fft, udy_fft

    @boost
    def project_kradial3d(self, vx_fft: Ac, vy_fft: Ac, vz_fft: Ac):
        r"""Project (inplace) a vector field parallel to the k-radial direction of the wavevector.

        Parameters
        ----------

        Arrays containing the velocity in Fourier space.

        Notes
        -----

        .. |kk| mathmacro:: \mathbf{k}
        .. |ee| mathmacro:: \mathbf{e}
        .. |vv| mathmacro:: \mathbf{v}

        The radial unitary vector for the mode :math:`\kk` is

        .. math::

           \ee_\kk = \frac{\kk}{|\kk|}
           = \sin \theta_\kk \cos \varphi_\kk ~ \ee_x
           + \sin \theta_\kk \sin \varphi_\kk ~ \ee_y
           + \cos \theta_\kk ~ \ee_z,

        and the projection of a velocity mode :math:`\hat{\vv}_\kk` along
        :math:`\ee_\kk` is

        .. math:: \hat{v}_\kk ~ \ee_\kk \equiv \hat{\vv}_\kk \cdot \ee_\kk ~ \ee_\kk

        This function set :math:`\hat{\vv}_\kk = \hat{v}_\kk ~ \ee_\kk` for all
        modes.

        .. note:

           For a divergent less vector field, the resulting vector is zero.

        """

        K_square_nozero = self.Kx**2 + self.Ky**2 + self.Kz**2
        K_square_nozero[K_square_nozero == 0] = 1e-14
        inv_K_square_nozero = 1.0 / K_square_nozero

        tmp = (
            self.Kx * vx_fft + self.Ky * vy_fft + self.Kz * vz_fft
        ) * inv_K_square_nozero

        n0, n1, n2 = vx_fft.shape
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    vx_fft[i0, i1, i2] = self.Kx[i0, i1, i2] * tmp[i0, i1, i2]
                    vy_fft[i0, i1, i2] = self.Ky[i0, i1, i2] * tmp[i0, i1, i2]
                    vz_fft[i0, i1, i2] = self.Kz[i0, i1, i2] * tmp[i0, i1, i2]

    @boost
    def project_poloidal(self, vx_fft: Ac, vy_fft: Ac, vz_fft: Ac):
        r"""Project (inplace) a vector field parallel to the k-poloidal (or polar) direction.

        Parameters
        ----------

        Arrays containing the velocity in Fourier space.

        Notes
        -----

        The poloidal unitary vector for the mode :math:`\kk` is

        .. math::

           \ee_{\kk\theta}
           = \cos \theta_\kk \cos \varphi_\kk ~ \ee_x
           + \cos \theta_\kk \sin \varphi_\kk ~ \ee_y - \sin \theta_\kk ~ \ee_z,

        and the projection of a velocity mode :math:`\hat{\vv}_\kk` along
        :math:`\ee_{\kk\theta}` is

        .. math::

           \hat{v}_{\kk\theta} ~ \ee_{\kk\theta}
           \equiv \hat{\vv}_\kk \cdot \ee_{\kk\theta} ~ \ee_{\kk\theta}

        This function set :math:`\hat{\vv}_\kk = \hat{v}_{\kk\theta} ~
        \ee_{\kk\theta}` for all modes.
        """

        Kh_square = self.Kx**2 + self.Ky**2
        K_square_nozero = Kh_square + self.Kz**2
        Kh_square_nozero = Kh_square.copy()

        Kh_square_nozero[Kh_square_nozero == 0] = 1e-14
        K_square_nozero[K_square_nozero == 0] = 1e-14

        inv_Kh_square_nozero = 1.0 / Kh_square_nozero
        inv_K_square_nozero = 1.0 / K_square_nozero

        cos_theta_k = self.Kz * np.sqrt(inv_K_square_nozero)
        sin_theta_k = np.sqrt(Kh_square * inv_K_square_nozero)
        cos_phi_k = self.Kx * np.sqrt(inv_Kh_square_nozero)
        sin_phi_k = self.Ky * np.sqrt(inv_Kh_square_nozero)

        tmp = (
            cos_theta_k * cos_phi_k * vx_fft
            + cos_theta_k * sin_phi_k * vy_fft
            - sin_theta_k * vz_fft
        )

        n0, n1, n2 = vx_fft.shape
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    vx_fft[i0, i1, i2] = (
                        cos_theta_k[i0, i1, i2]
                        * cos_phi_k[i0, i1, i2]
                        * tmp[i0, i1, i2]
                    )
                    vy_fft[i0, i1, i2] = (
                        cos_theta_k[i0, i1, i2]
                        * sin_phi_k[i0, i1, i2]
                        * tmp[i0, i1, i2]
                    )
                    vz_fft[i0, i1, i2] = (
                        -sin_theta_k[i0, i1, i2] * tmp[i0, i1, i2]
                    )

    @boost
    def vpfft_from_vecfft(self, vx_fft: Ac, vy_fft: Ac, vz_fft: Ac):
        """Return the poloidal component of a vector field."""

        Kh_square = self.Kx**2 + self.Ky**2
        K_square_nozero = Kh_square + self.Kz**2
        Kh_square_nozero = Kh_square.copy()

        Kh_square_nozero[Kh_square_nozero == 0] = 1e-14
        K_square_nozero[K_square_nozero == 0] = 1e-14

        inv_Kh_square_nozero = 1.0 / Kh_square_nozero
        inv_K_square_nozero = 1.0 / K_square_nozero

        cos_theta_k = self.Kz * np.sqrt(inv_K_square_nozero)
        sin_theta_k = np.sqrt(Kh_square * inv_K_square_nozero)
        cos_phi_k = self.Kx * np.sqrt(inv_Kh_square_nozero)
        sin_phi_k = self.Ky * np.sqrt(inv_Kh_square_nozero)

        result = (
            cos_theta_k * cos_phi_k * vx_fft
            + cos_theta_k * sin_phi_k * vy_fft
            - sin_theta_k * vz_fft
        )

        return result

    @boost
    def vecfft_from_vpfft(self, vp_fft: Ac):
        """Return a vector field from the poloidal component."""

        Kh_square = self.Kx**2 + self.Ky**2
        K_square_nozero = Kh_square + self.Kz**2
        Kh_square_nozero = Kh_square.copy()

        Kh_square_nozero[Kh_square_nozero == 0] = 1e-14
        K_square_nozero[K_square_nozero == 0] = 1e-14

        inv_Kh_square_nozero = 1.0 / Kh_square_nozero
        inv_K_square_nozero = 1.0 / K_square_nozero

        cos_theta_k = self.Kz * np.sqrt(inv_K_square_nozero)
        sin_theta_k = np.sqrt(Kh_square * inv_K_square_nozero)
        cos_phi_k = self.Kx * np.sqrt(inv_Kh_square_nozero)
        sin_phi_k = self.Ky * np.sqrt(inv_Kh_square_nozero)

        ux_fft = cos_theta_k * cos_phi_k * vp_fft
        uy_fft = cos_theta_k * sin_phi_k * vp_fft
        uz_fft = -sin_theta_k * vp_fft

        return ux_fft, uy_fft, uz_fft

    @boost
    def project_toroidal(self, vx_fft: Ac, vy_fft: Ac, vz_fft: Ac):
        r"""Project (inplace) a vector field parallel to the k-toroidal (or azimutal) direction.

        Parameters
        ----------

        Arrays containing the velocity in Fourier space.

        Notes
        -----

        The toroidal unitary vector for the mode :math:`\kk` is

        .. math::

           \ee_{\kk\varphi}
           = - \sin \varphi_\kk ~ \ee_x + \cos \varphi_\kk ~ \mathbb{e}_y,

        and the projection of a velocity mode :math:`\hat{\vv}_\kk` along
        :math:`\ee_{\kk\varphi}` is

        .. math::

           \hat{v}_{\kk\varphi} ~ \ee_{\kk\varphi}
           \equiv \hat{\vv}_\kk \cdot \ee_{\kk\varphi} ~ \ee_{\kk\varphi}

        This function compute :math:`\hat{\vv}_\kk = \hat{v}_{\kk\varphi} ~
        \ee_{\kk\varphi}` for all modes.
        """

        Kh_square_nozero = self.Kx**2 + self.Ky**2
        Kh_square_nozero[Kh_square_nozero == 0] = 1e-14
        inv_Kh_square_nozero = 1.0 / Kh_square_nozero
        del Kh_square_nozero

        tmp = np.sqrt(inv_Kh_square_nozero)
        cos_phi_k = self.Kx * tmp
        sin_phi_k = self.Ky * tmp

        tmp = -sin_phi_k * vx_fft + cos_phi_k * vy_fft

        n0, n1, n2 = vx_fft.shape
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    vx_fft[i0, i1, i2] = -sin_phi_k[i0, i1, i2] * tmp[i0, i1, i2]
                    vy_fft[i0, i1, i2] = cos_phi_k[i0, i1, i2] * tmp[i0, i1, i2]
                    vz_fft[i0, i1, i2] = 0.0

    @boost
    def vtfft_from_vecfft(self, vx_fft: Ac, vy_fft: Ac, vz_fft: Ac):
        """Return the toroidal component of a vector field."""

        Kh_square_nozero = self.Kx**2 + self.Ky**2
        Kh_square_nozero[Kh_square_nozero == 0] = 1e-14
        inv_Kh_square_nozero = 1.0 / Kh_square_nozero
        del Kh_square_nozero

        tmp = np.sqrt(inv_Kh_square_nozero)
        cos_phi_k = self.Kx * tmp
        sin_phi_k = self.Ky * tmp

        result = -sin_phi_k * vx_fft + cos_phi_k * vy_fft

        return result

    @boost
    def vecfft_from_vtfft(self, vt_fft: Ac):
        """Return a 3D vector field from the toroidal component."""

        Kh_square = self.Kx**2 + self.Ky**2
        K_square_nozero = Kh_square + self.Kz**2
        Kh_square_nozero = Kh_square.copy()

        Kh_square_nozero[Kh_square_nozero == 0] = 1e-14
        K_square_nozero[K_square_nozero == 0] = 1e-14

        inv_Kh_square_nozero = 1.0 / Kh_square_nozero

        cos_phi_k = self.Kx * np.sqrt(inv_Kh_square_nozero)
        sin_phi_k = self.Ky * np.sqrt(inv_Kh_square_nozero)

        ux_fft = -sin_phi_k * vt_fft
        uy_fft = cos_phi_k * vt_fft
        uz_fft = 0.0 * vt_fft

        return ux_fft, uy_fft, uz_fft

    @boost
    def divhfft_from_vxvyfft(self, vx_fft: Ac, vy_fft: Ac):
        """Compute the horizontal divergence in spectral space."""
        return 1j * (self.Kx * vx_fft + self.Ky * vy_fft)

    @boost
    def vxvyfft_from_rotzfft(self, rotz_fft: Ac):

        inv_Kh_square_nozero = self.Kx**2 + self.Ky**2
        inv_Kh_square_nozero[inv_Kh_square_nozero == 0] = 1e-14
        inv_Kh_square_nozero = 1 / inv_Kh_square_nozero

        vx_fft = 1j * self.Ky * inv_Kh_square_nozero * rotz_fft
        vy_fft = -1j * self.Kx * inv_Kh_square_nozero * rotz_fft
        return vx_fft, vy_fft

    @boost
    def vxvyfft_from_divhfft(self, divh_fft: Ac):

        inv_Kh_square_nozero = self.Kx**2 + self.Ky**2
        inv_Kh_square_nozero[inv_Kh_square_nozero == 0] = 1e-14
        inv_Kh_square_nozero = 1 / inv_Kh_square_nozero

        vx_fft = -1j * self.Kx * inv_Kh_square_nozero * divh_fft
        vy_fft = -1j * self.Ky * inv_Kh_square_nozero * divh_fft
        return vx_fft, vy_fft

    def get_grid1d_seq(self, axe="x"):

        if axe not in ("x", "y", "z"):
            raise ValueError

        if self.params.ONLY_COARSE_OPER:
            number_points = getattr(self.params.oper, "n" + axe)
            length = getattr(self, "L" + axe)
            return np.linspace(0, length, number_points)
        else:
            return getattr(self, axe + "_seq")

    def project_fft_on_realX(self, f_fft):
        return self.fft(self.ifft(f_fft))

    def _ikxyzseq_from_ik012rank(self, ik0, ik1, ik2, rank=0):
        """Give the sequential indices (xyz) from the local indices and the rank"""
        if self._is_mpi_lib:
            # much more complicated in this case
            raise NotImplementedError
        dimX_K = self.oper_fft.get_dimX_K()
        if dimX_K == (0, 1, 2):
            ikz, iky, ikx = ik0, ik1, ik2
        else:
            raise NotImplementedError(
                f"dimX_K={dimX_K} not implemented ({self.oper_fft.__class__})"
            )
        return ikx, iky, ikz

    def _ik012rank_from_ikxyzseq(self, ikx, iky, ikz):
        """Give the local indices and the rank from "sequential" indices (xyz)"""
        if self._is_mpi_lib:
            # much more complicated in this case
            raise NotImplementedError
        rank_k = 0
        dimX_K = self.oper_fft.get_dimX_K()
        if dimX_K == (0, 1, 2):
            ik0, ik1, ik2 = ikz, iky, ikx
        else:
            raise NotImplementedError
        return ik0, ik1, ik2, rank_k

    def _kadim_from_ikxyzseq(self, ikx, iky, ikz):
        """Give the adimensional wavenumbers from sequential indices"""
        kx_adim = ikx
        ky_adim = _kadim_from_ik(iky, self.ny)
        kz_adim = _kadim_from_ik(ikz, self.nz)
        return kx_adim, ky_adim, kz_adim

    def _ikxyzseq_from_kadim(self, kx_adim, ky_adim, kz_adim):
        """Give the sequential indices from the adimensional wavenumbers"""
        ikx = kx_adim
        iky = _ik_from_kadim(ky_adim, self.ny)
        ikz = _ik_from_kadim(kz_adim, self.nz)
        return ikx, iky, ikz

    def kadim_from_ik012rank(self, ik0, ik1, ik2, rank=0):
        """Give the adimensional wavenumbers from local indices and rank"""
        ikx, iky, ikz = self._ikxyzseq_from_ik012rank(ik0, ik1, ik2, rank)
        return self._kadim_from_ikxyzseq(ikx, iky, ikz)

    def ik012rank_from_kadim(self, kx_adim, ky_adim, kz_adim):
        """Give the local indices and rank from the adimensional wavenumbers"""
        ikx, iky, ikz = self._ikxyzseq_from_kadim(kx_adim, ky_adim, kz_adim)
        return self._ik012rank_from_ikxyzseq(ikx, iky, ikz)

    def set_value_spect(
        self, arr_fft, value, kx_adim, ky_adim, kz_adim, from_rank=0
    ):
        """Set a value in a spectral array given the adimensional wavenumber"""
        ik0, ik1, ik2, rank_k = self.ik012rank_from_kadim(
            kx_adim, ky_adim, kz_adim
        )
        if mpi.rank != rank_k or from_rank != 0:
            raise NotImplementedError
        # print("-" * 20)
        # print(f"ik0, ik1, ik2             = ({ik0:4d}, {ik1:4d}, {ik2:4d})")
        arr_fft[ik0, ik1, ik2] = value

    def get_value_spect(self, arr_fft, kx_adim, ky_adim, kz_adim, to_rank=0):
        """Get a value in a spectral array given the adimensional wavenumber"""
        ik0, ik1, ik2, rank_k = self.ik012rank_from_kadim(
            kx_adim, ky_adim, kz_adim
        )
        if mpi.rank != rank_k or to_rank != 0:
            raise NotImplementedError
        return arr_fft[ik0, ik1, ik2]

    @boost
    def get_phases_random(self):
        # Not supported by Pythran 0.9.5!
        # alpha_x, alpha_y, alpha_z = np.random.uniform(-0.5, 0.5, 3)
        alpha_x, alpha_y, alpha_z = tuple(uniform(-0.5, 0.5) for _ in range(3))
        beta_x = alpha_x + 0.5 if alpha_x < 0 else alpha_x - 0.5
        beta_y = alpha_y + 0.5 if alpha_y < 0 else alpha_y - 0.5
        beta_z = alpha_z + 0.5 if alpha_z < 0 else alpha_z - 0.5

        phase_alpha = (
            alpha_x * self.deltax * self.Kx
            + alpha_y * self.deltay * self.Ky
            + alpha_z * self.deltaz * self.Kz
        )
        phase_beta = (
            beta_x * self.deltax * self.Kx
            + beta_y * self.deltay * self.Ky
            + beta_z * self.deltaz * self.Kz
        )
        return phase_alpha, phase_beta


def _ik_from_ikc(ikc, nkc, nk):
    if ikc <= nkc / 2.0:
        ik = ikc
    else:
        knodim = ikc - nkc
        ik = knodim + nk
    return ik


def _kadim_from_ik(ik, nk, first=False):
    if first or ik <= nk // 2:
        return ik
    return ik - nk


def _ik_from_kadim(k_adim, nk, first=False):
    if first or k_adim >= 0:
        return k_adim
    return nk + k_adim


if __name__ == "__main__":
    n = 4

    p = OperatorsPseudoSpectral3D._create_default_params()

    p.oper.nx = n
    p.oper.ny = 2 * n
    p.oper.nz = 4 * n

    # p.oper.type_fft = 'fftwpy'
    p.oper.type_fft2d = "fft2d.with_pyfftw"

    oper = OperatorsPseudoSpectral3D(params=p)

    field = np.ones(oper.shapeX_loc)

    print(oper.shapeX_loc)
    print(oper.oper2d.shapeX_loc)

    field_fft = oper.fft3d(field)

    assert field_fft.shape == oper.shapeK_loc

    oper.project_perpk3d(field_fft, field_fft, field_fft)

    a2d = np.arange(oper.nx * oper.ny).reshape(oper.oper2d.shapeX_loc)
    a3d = oper.build_invariant_arrayX_from_2d_indices12X(a2d)
