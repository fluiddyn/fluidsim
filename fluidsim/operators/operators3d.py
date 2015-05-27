
import sys

import numpy as np

from math import pi

from fluidsim.operators.fft.easypyfft import FFTW3DReal2Complex
from fluiddyn.util.mpi import nb_proc
from fluidsim.operators.operators import OperatorsPseudoSpectral2D
from fluidsim.base.setofvariables import SetOfVariables


def _make_str_length(length):
    if (length/np.pi).is_integer():
        return repr(int(length/np.pi)) + 'pi'
    else:
        return '{:.3f}'.format(length).rstrip('0')


class OperatorsPseudoSpectral3D(object):
    """Provides fast Fourier transform functions and 3D operators.

    """

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """

        if nb_proc > 1:
            type_fft = 'FFTWCCY'
        else:
            if not sys.platform == 'win32':
                type_fft = 'FFTWCY'
            else:
                type_fft = 'FFTWPY'

        attribs = {'type_fft': type_fft,
                   'TRANSPOSED_OK': True,
                   'coef_dealiasing': 2./3,
                   'nx': 48,
                   'ny': 48,
                   'nz': 48,
                   'Lx': 2*pi,
                   'Ly': 2*pi,
                   'Lz': 2*pi}
        params._set_child('oper', attribs=attribs)

    def __init__(self, params=None, SEQUENTIAL=None):

        self.params = params

        nx = self.nx_seq = params.oper.nx
        ny = self.ny_seq = params.oper.ny
        nz = self.nz_seq = params.oper.nz

        if nx % 2 != 0 or ny % 2 != 0 or nz % 2 != 0:
            raise ValueError('nx, ny and nz have to be even.')

        self.shape_phys = (nz, ny, nx)
        self.shapeX_loc = self.shape_phys

        Lx = self.Lx = params.oper.Lx
        Ly = self.Ly = params.oper.Ly
        Lz = self.Lz = params.oper.Lz

        self.deltax = Lx/nx
        self.deltay = Ly/ny
        self.deltaz = Lz/nz

        self._oper2d = OperatorsPseudoSpectral2D(params)
        self.type_fft = self._oper2d.type_fft
        self.ifft2 = self.ifft2d = self._oper2d.ifft2
        self.fft2 = self.fft2d = self._oper2d.fft2

        self._op_fft = FFTW3DReal2Complex(nx, ny, nz)

        self.ifft3d = self._op_fft.ifft3d
        self.fft3d = self._op_fft.fft3d
        self.sum_wavenumbers = self._op_fft.sum_wavenumbers

        kx_adim_max = nx/2
        ky_adim_max = ny/2
        kz_adim_max = nz/2

        self.nkx = kx_adim_max + 1
        self.nky = ny
        self.nkz = nz

        self.nk0 = self.nkz
        self.nk1 = self.nky
        self.nk2 = self.nkx

        self.shapeK_loc = (self.nk0, self.nk1, self.nk2)

        self.deltakx = 2*pi/Lx
        self.deltaky = 2*pi/Ly
        self.deltakz = 2*pi/Lz

        self.k0 = self.deltakz * np.r_[0:kz_adim_max+1, -kz_adim_max+1:0]
        self.k1 = self.deltakz * np.r_[0:ky_adim_max+1, -ky_adim_max+1:0]
        self.k2 = self.deltakx * np.arange(self.nk2)

        K1, K0, K2 = np.meshgrid(self.k1, self.k0, self.k2, copy=False)

        self.Kz = K0
        self.Ky = K1
        self.Kx = K2

        self.K2 = K0**2 + K1**2 + K2**2
        self.K8 = self.K2**4

        self.K_square_nozero = self.K2.copy()
        self.K_square_nozero[0, 0, 0] = 1e-14

        self.coef_dealiasing = params.oper.coef_dealiasing

        CONDKX = abs(self.Kx) > self.coef_dealiasing*self.k2.max()
        CONDKY = abs(self.Ky) > self.coef_dealiasing*self.k1.max()
        CONDKZ = abs(self.Kz) > self.coef_dealiasing*self.k0.max()
        where_dealiased = np.logical_or(CONDKX, CONDKY, CONDKZ)
        self.where_dealiased = np.array(where_dealiased, dtype=np.int8)

    def produce_str_describing_oper(self):
        """Produce a string describing the operator."""
        str_Lx = _make_str_length(self.Lx)
        str_Ly = _make_str_length(self.Ly)
        str_Lz = _make_str_length(self.Lz)

        return ('L=' + str_Lx + 'x' + str_Ly + 'x' + str_Lz +
                '_{}x{}x{}').format(self.nx_seq, self.ny_seq, self.nz_seq)

    def produce_long_str_describing_oper(self):
        """Produce a string describing the operator."""

        str_Lx = _make_str_length(self.Lx)
        str_Ly = _make_str_length(self.Ly)
        str_Lz = _make_str_length(self.Lz)

        return (
            'type fft: ' + self.type_fft + '\n' +
            'nx = {0:6d} ; ny = {1:6d}\n'.format(self.nx_seq, self.ny_seq) +
            'Lx = ' + str_Lx + ' ; Ly = ' + str_Ly +
            ' ; Lz = ' + str_Lz + '\n')

    def expand_3dfrom2d(self, arr2d):
        return arr2d.repeat(self.nz_seq).reshape((self.nz_seq,) + arr2d.shape)

    def project_perpk3d(self, vx_fft, vy_fft, vz_fft):

        Kx = self.Kx
        Ky = self.Ky
        Kz = self.Kz

        tmp = (Kx * vx_fft + Ky * vy_fft + Kz * vz_fft) / self.K_square_nozero

        return (vx_fft - Kx * tmp, vy_fft - Ky * tmp, vz_fft - Kz * tmp)

    def vgradv_from_v(self, vx, vy, vz, vx_fft=None, vy_fft=None, vz_fft=None):

        ifft3d = self.ifft3d

        if vx_fft is None:
            vx_fft = self.fft3d(vx)
            vy_fft = self.fft3d(vy)
            vz_fft = self.fft3d(vz)

        Kx = self.Kx
        Ky = self.Ky
        Kz = self.Kz

        px_vx_fft = 1j * Kx * vx_fft
        py_vx_fft = 1j * Ky * vx_fft
        pz_vx_fft = 1j * Kz * vx_fft

        px_vy_fft = 1j * Kx * vy_fft
        py_vy_fft = 1j * Ky * vy_fft
        pz_vy_fft = 1j * Kz * vy_fft

        px_vz_fft = 1j * Kx * vz_fft
        py_vz_fft = 1j * Ky * vz_fft
        pz_vz_fft = 1j * Kz * vz_fft

        vgradvx = (vx * ifft3d(px_vx_fft) +
                   vy * ifft3d(py_vx_fft) +
                   vz * ifft3d(pz_vx_fft))

        vgradvy = (vx * ifft3d(px_vy_fft) +
                   vy * ifft3d(py_vy_fft) +
                   vz * ifft3d(pz_vy_fft))

        vgradvz = (vx * ifft3d(px_vz_fft) +
                   vy * ifft3d(py_vz_fft) +
                   vz * ifft3d(pz_vz_fft))

        return vgradvx, vgradvy, vgradvz

    def dealiasing(self, *arguments):
        for ii in range(len(arguments)):
            thing = arguments[ii]
            if isinstance(thing, SetOfVariables):
                _dealiasing_setofvar(thing, self.where_dealiased)
            elif isinstance(thing, np.ndarray):
                _dealiasing_variable(thing, self.where_dealiased)


def _dealiasing_setofvar(sov, where_dealiased):
    for i in range(sov.shape[0]):
        sov[i][where_dealiased] = 0.


def _dealiasing_variable(ff_fft, where_dealiased):
    ff_fft[where_dealiased] = 0.


if __name__ == '__main__':
    n = 4

    from fluidsim.base.params import Parameters
    p = Parameters(tag='params', attribs={'ONLY_COARSE_OPER': False})
    p._set_child(
        'oper', {'nx': n, 'ny': n, 'nz': 2*n,
                 'Lx': 2*pi, 'Ly': 2*pi, 'Lz': 2*pi,
                 'type_fft': 'FFTWPY', 'coef_dealiasing': 0.66,
                 'TRANSPOSED_OK': True})

    oper = OperatorsPseudoSpectral3D(params=p)

    field = np.ones(oper.shape_phys)

    field_fft = oper.fft3d(field)

    assert field_fft.shape == oper.shapeK_loc

    oper.vgradv_from_v(field, field, field)

    oper.project_perpk3d(field_fft, field_fft, field_fft)

    a2d = np.arange(oper.nx*oper.ny).reshape([oper.ny, oper.nx])
    a3d = oper.expand_3dfrom2d(a2d)
