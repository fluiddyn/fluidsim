"""Operators 3d (:mod:`fluidsim.operators.operators3d`)
=======================================================

Provides

.. autoclass:: OperatorsPseudoSpectral3D
   :members:
   :private-members:

"""

from __future__ import division

from builtins import range

from math import pi
from copy import deepcopy

import numpy as np

from fluiddyn.util.mpi import nb_proc, rank

from fluidsim.base.setofvariables import SetOfVariables

from fluidfft.fft3d.operators import OperatorsPseudoSpectral3D as _Operators

from .operators2d import OperatorsPseudoSpectral2D as OpPseudoSpectral2D

from . import util3d_pythran


def dealiasing_setofvar_numpy(sov, where_dealiased):
    for i in range(sov.shape[0]):
        sov[i][np.nonzero(where_dealiased)] = 0.


def dealiasing_variable_numpy(ff_fft, where_dealiased):
    ff_fft[np.nonzero(where_dealiased)] = 0.


if hasattr(util3d_pythran, "__pythran__"):
    dealiasing_variable = util3d_pythran.dealiasing_variable
    dealiasing_setofvar = util3d_pythran.dealiasing_setofvar
else:
    dealiasing_variable = dealiasing_variable_numpy
    dealiasing_setofvar = dealiasing_setofvar_numpy


class OperatorsPseudoSpectral3D(_Operators):
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

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """

        if nb_proc > 1:
            type_fft = "fft3d.mpi_with_fftwmpi3d"
        else:
            type_fft = "fft3d.with_pyfftw"

        attribs = {
            "type_fft": type_fft,
            "type_fft2d": "fft2d.with_pyfftw",
            "TRANSPOSED_OK": True,
            "coef_dealiasing": 2. / 3,
            "nx": 48,
            "ny": 48,
            "nz": 48,
            "Lx": 2 * pi,
            "Ly": 2 * pi,
            "Lz": 2 * pi
        }
        params._set_child("oper", attribs=attribs)

    def __init__(self, params=None):

        self.params = params

        super(OperatorsPseudoSpectral3D, self).__init__(
            params.oper.nx,
            params.oper.ny,
            params.oper.nz,
            params.oper.Lx,
            params.oper.Ly,
            params.oper.Lz,
            fft=params.oper.type_fft,
            coef_dealiasing=params.oper.coef_dealiasing,
        )

        # problem here type_fft
        params2d = deepcopy(params)
        params2d.oper.type_fft = params2d.oper.type_fft2d
        fft = params2d.oper.type_fft

        if any([fft.startswith(s) for s in ["fluidfft.fft2d.", "fft2d."]]):
            self.oper2d = OpPseudoSpectral2D(params2d)
        else:
            raise ValueError

        self.ifft2 = self.ifft2d = self.oper2d.ifft2
        self.fft2 = self.fft2d = self.oper2d.fft2

    def build_invariant_arrayX_from_2d_indices12X(self, arr2d):
        """Build a 3D array from a 2D array"""
        return self._op_fft.build_invariant_arrayX_from_2d_indices12X(
            self.oper2d, arr2d
        )

    def build_invariant_arrayK_from_2d_indices12X(self, arr2d):
        """Build a 3D array from a 2D array"""
        return self._op_fft.build_invariant_arrayK_from_2d_indices12X(
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
            raise NotImplementedError

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
            raise NotImplementedError()

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

    def urudfft_from_vxvyfft(self, vx_fft, vy_fft):
        """Compute toroidal and poloidal horizontal velocities

        """
        return util3d_pythran.urudfft_from_vxvyfft(
            vx_fft, vy_fft, self.Kx, self.Ky, rank
        )


def _ik_from_ikc(ikc, nkc, nk):
    if ikc <= nkc / 2.:
        ik = ikc
    else:
        knodim = ikc - nkc
        ik = knodim + nk
    return ik


if __name__ == "__main__":
    n = 4

    from fluidsim.base.params import Parameters

    p = Parameters(tag="params", attribs={"ONLY_COARSE_OPER": False})
    OperatorsPseudoSpectral3D._complete_params_with_default(p)

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

    oper.vgradv_from_v(field, field, field)

    oper.project_perpk3d(field_fft, field_fft, field_fft)

    a2d = np.arange(oper.nx * oper.ny).reshape(oper.oper2d.shapeX_loc)
    a3d = oper.build_invariant_arrayX_from_2d_indices12X(a2d)
