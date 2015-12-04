

from math import pi

import numpy as np


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


    Uses fft operators that implement the methods:

    - ifft
    - fft
    - get_shapeX_loc
    - get_shapeX_seq
    - get_shapeK_loc
    - get_shapeK_seq
    - get_orderK_dimX
    - get_seq_index_firstK

    - get_k_adim_loc
    - sum_wavenumbers

    """

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """

        if nb_proc > 1:
            type_fft = 'FluidPFFT'
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

        self.type_fft = type_fft = params.oper.type_fft
        if type_fft == 'FFTWPY':
            op_fft = self._op_fft = FFTW3DReal2Complex(nx, ny, nz)
        else:
            raise NotImplementedError

        # there is a problem here type_fft
        self._oper2d = OperatorsPseudoSpectral2D(params)
        self.ifft2 = self.ifft2d = self._oper2d.ifft2
        self.fft2 = self.fft2d = self._oper2d.fft2

        self.shapeX_seq = op_fft.get_shapeX_seq()
        self.shapeX_loc = op_fft.get_shapeX_loc()

        Lx = self.Lx = params.oper.Lx
        Ly = self.Ly = params.oper.Ly
        Lz = self.Lz = params.oper.Lz

        self.deltax = Lx/nx
        self.deltay = Ly/ny
        self.deltaz = Lz/nz

        self.x_seq = self.deltax*np.arange(nx)
        self.y_seq = self.deltay*np.arange(ny)
        self.z_seq = self.deltaz*np.arange(nz)

        deltakx = 2*pi/Lx
        deltaky = 2*pi/Ly
        deltakz = 2*pi/Lz

        self.ifft3d = op_fft.ifft
        self.fft3d = op_fft.fft
        self.sum_wavenumbers = op_fft.sum_wavenumbers

        self.shapeK_loc = op_fft.get_shapeK_loc()
        self.nk0, self.nk1, self.nk2 = self.shapeK_loc

        order = op_fft.get_orderK_dimX()
        if order == (0, 1, 2):
            self.deltaks = deltakz, deltaky, deltakx
        elif order == (1, 0, 2):
            self.deltaks = deltaky, deltakz, deltakx
        elif order == (2, 1, 0):
            self.deltaks = deltakx, deltaky, deltakz
        else:
            raise NotImplementedError

        k0_adim_loc, k1_adim_loc, k2_adim_loc = op_fft.get_k_adim_loc()

        self.k0 = self.deltaks[0] * k0_adim_loc
        self.k1 = self.deltaks[1] * k1_adim_loc
        self.k2 = self.deltaks[2] * k2_adim_loc

        # oh that's strange!
        K1, K0, K2 = np.meshgrid(self.k1, self.k0, self.k2, copy=False)

        self.Kz = K0
        self.Ky = K1
        self.Kx = K2

        self.K2 = K0**2 + K1**2 + K2**2
        self.K8 = self.K2**4

        self.seq_index_firstK0, self.seq_index_firstK1 = \
            op_fft.get_seq_index_firstK()

        self.K_square_nozero = self.K2.copy()

        if self.seq_index_firstK0 == 0 and self.seq_index_firstK1 == 0:
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
        if arr2d.dtype == np.complex128:
            ret = np.zeros((self.nz_seq,) + arr2d.shape, dtype=np.complex128)
            ret[0] = arr2d
            return ret
        else:
            return np.array(list(arr2d)*self.nz_seq).reshape(
                (self.nz_seq,) + arr2d.shape)

    def project_perpk3d(self, vx_fft, vy_fft, vz_fft):

        Kx = self.Kx
        Ky = self.Ky
        Kz = self.Kz

        tmp = (Kx * vx_fft + Ky * vy_fft + Kz * vz_fft) / self.K_square_nozero

        return (vx_fft - Kx * tmp,
                vy_fft - Ky * tmp,
                vz_fft - Kz * tmp)

    def vgradv_from_v(self, vx, vy, vz, vx_fft=None, vy_fft=None, vz_fft=None):

        ifft3d = self.ifft3d

        if vx_fft is None:
            fft3d = self.fft3d
            vx_fft = fft3d(vx)
            vy_fft = fft3d(vy)
            vz_fft = fft3d(vz)

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

    def dealiasing(self, *args):
        for thing in args:
            if isinstance(thing, SetOfVariables):
                _dealiasing_setofvar(thing, self.where_dealiased)
            elif isinstance(thing, np.ndarray):
                _dealiasing_variable(thing, self.where_dealiased)


def _dealiasing_setofvar(sov, where_dealiased):
    for i in range(sov.shape[0]):
        sov[i][np.nonzero(where_dealiased)] = 0.


def _dealiasing_variable(ff_fft, where_dealiased):
    ff_fft[np.nonzero(where_dealiased)] = 0.


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
