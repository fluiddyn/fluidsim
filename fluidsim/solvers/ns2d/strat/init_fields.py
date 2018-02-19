"""Initialization of the field (:mod:`fluidsim.solvers.ns2d.strat.init_fields`)
===============================================================================

.. autoclass:: InitFieldsNS2DStrat
   :members:

.. autoclass:: InitFieldsNoiseStrat
   :members:

.. autoclass:: InitFieldsJetStrat
   :members:

.. autoclass:: InitFieldsDipoleStrat
   :members:

"""

from __future__ import print_function

import numpy as np

from fluiddyn.util import mpi
from past.utils import old_div

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields
from fluidsim.solvers.ns2d.init_fields import InitFieldsNoise, \
    InitFieldsJet, InitFieldsDipole

InitFieldsJetStrat = InitFieldsJet
InitFieldsDipoleStrat = InitFieldsDipole

# It would be nice that rot_fft, ux_fft and uy_fft are attributes of the class
# init_fields.


class InitFieldsNoiseStrat(InitFieldsNoise):
    """
    Class to initialize the fields with noise for the solver ns2d.strat.

    """

    def compute_rotuxuy_fft(self):
        """
        Compute the rot_fft, ux_fft and uy_fft from a random noise field.
        """
        params = self.sim.params
        oper = self.sim.oper

        lambda0 = params.init_fields.noise.length
        if lambda0 == 0:
            lambda0 = old_div(oper.Lx, 4)

        def H_smooth(x, delta):
            return old_div((1. + np.tanh(2*np.pi*x/delta)), 2)

        # to compute always the same field... (for 1 resolution...)
        np.random.seed(42)  # this does not work for MPI...

        ux_fft = (np.random.random(oper.shapeK) +
                  1j*np.random.random(oper.shapeK) - 0.5 - 0.5j)
        uy_fft = (np.random.random(oper.shapeK) +
                  1j*np.random.random(oper.shapeK) - 0.5 - 0.5j)

        if mpi.rank == 0:
            ux_fft[0, 0] = 0.
            uy_fft[0, 0] = 0.

        oper.projection_perp(ux_fft, uy_fft)
        oper.dealiasing(ux_fft, uy_fft)

        k0 = 2*np.pi/lambda0
        delta_k0 = 1.*k0
        ux_fft = ux_fft*H_smooth(k0-oper.KK, delta_k0)
        uy_fft = uy_fft*H_smooth(k0-oper.KK, delta_k0)

        ux = oper.ifft2(ux_fft)
        uy = oper.ifft2(uy_fft)
        velo_max = np.sqrt(ux**2+uy**2).max()
        if mpi.nb_proc > 1:
            velo_max = oper.comm.allreduce(velo_max, op=mpi.MPI.MAX)
        ux = params.init_fields.noise.velo_max*ux/velo_max
        uy = params.init_fields.noise.velo_max*uy/velo_max
        ux_fft = oper.fft2(ux)
        uy_fft = oper.fft2(uy)

        # if NO_SHEAR_MODES --> No energy in shear modes!
        if self.sim.params.NO_SHEAR_MODES:
            ux_fft[:, 0] = 0
            uy_fft[:, 0] = 0

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        return rot_fft, ux_fft, uy_fft


class InitFieldsLinearMode(SpecificInitFields):
    """
    Class to initialize the fields with the linear mode
    """
    tag = 'linear_mode'

    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the `params` container (class method)."""
        super(InitFieldsLinearMode, cls)._complete_params_with_default(params)
        params.init_fields._set_child(cls.tag, attribs={
            'i_mode': (8, 8),
            'delta_k_adim': 1,
            'amplitude': 1})

        params.init_fields.linear_mode._set_doc("""
        i_mode : tuple

          Index of initialized mode.

         delta_k_adim : int

          Size of the initialization region.

        amplitude : float

          Amplitude of the initial linear mode

        """)


    def __call__(self):
        if mpi.nb_proc > 1:
            raise NotImplementedError(
                'Function compute_apfft_ones not implemented in MPI.')

        ap_fft = self.compute_apfft_ones()
        self.sim.state.init_statespect_from(ap_fft=ap_ffr)
        
    def compute_apfft_ones(self):
        """Compute the linear mode apfft"""

        params = self.sim.params
        oper = self.sim.oper

        i_mode = params.init_fields.linear_mode.i_mode
        delta_k_adim = params.init_fields.linear_mode.delta_k_adim
        amplitude = params.init_fields.linear_mode.amplitude * params.N**2

        am_fft = np.zeros(oper.shapeK) + 1j * np.zeros(oper.shapeK)
        am_fft = amplitude * am_fft
        ap_fft = am_fft.copy()

        # Define contour delta_k_adim
        # region [epsx_min: epsx_max, epsy_min, epsymax]
        epsx_min = i_mode[0] - delta_k_adim
        epsy_min = i_mode[1] - delta_k_adim

        epsx_max = i_mode[0] + delta_k_adim + 1
        epsy_max = i_mode[1] + delta_k_adim + 1

        if epsx_min < 0:
            epsx_min = 0
        if epsy_min < 0:
            epsy_min = 0

        ap_fft[epsx_min:epsx_max, epsy_min:epsy_max] = 1. + 0j
        ap_fft = amplitude * ap_fft
        return ap_fft


class InitFieldsNS2DStrat(InitFieldsBase):
    """Initialize the state for the solver ns2d.strat."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""
        InitFieldsBase._complete_info_solver(
            info_solver, classes=[InitFieldsNoiseStrat, InitFieldsJetStrat,
                                  InitFieldsDipoleStrat, InitFieldsLinearMode])
