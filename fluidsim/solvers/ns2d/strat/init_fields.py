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

import random
import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields
from fluidsim.solvers.ns2d.init_fields import (
    InitFieldsNoise,
    InitFieldsJet,
    InitFieldsDipole,
)

InitFieldsJetStrat = InitFieldsJet
InitFieldsDipoleStrat = InitFieldsDipole

# It would be nice that rot_fft, ux_fft and uy_fft are attributes of the class
# init_fields.


class InitFieldsNoiseStrat(InitFieldsNoise):
    """
    Class to initialize the fields with noise for the solver ns2d.strat.

    """


class InitFieldsLinearMode(SpecificInitFields):
    """
    Class to initialize the fields with the linear mode
    """

    tag = "linear_mode"

    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the `params` container (class method)."""
        super()._complete_params_with_default(params)
        p_linear_mode = params.init_fields._set_child(
            cls.tag,
            attribs={
                "i_mode": (8, 8),
                "delta_k_adim": 1,
                "amplitude": 1,
                "eigenmode": "ap_fft",
            },
        )

        p_linear_mode._set_doc(
            """
        i_mode : tuple

            Index of initialized mode (kx, kz).

        delta_k_adim : int

            Size (in modes) of the initialized region.
            Example:
             delta_k_adim = 1 : 1 mode initialized
             delta_k_adim = 2 : 4 modes initialized

        amplitude : float

          Amplitude of the initial linear mode

        eigenmode : str

          Which eigenmode (ap_fft or am_fft)

        """
        )

    def __call__(self):

        params = self.sim.params

        linear_mode = self.put_ones_linear_mode()

        eigenmode = self.sim.params.init_fields.linear_mode.eigenmode

        if eigenmode == "ap_fft":
            print("hello")
            self.sim.state.init_statespect_from(ap_fft=linear_mode)
        elif eigenmode == "am_fft":
            print("Is here?")
            self.sim.state.init_statespect_from(am_fft=linear_mode)
        else:
            raise ValueError("eigenmode should be ap_fft or am_fft.")

    def put_ones_linear_mode(self):
        """Put ones in the linear mode ap_fft or am_fft"""

        oper = self.sim.oper
        params = self.sim.params

        i_mode = params.init_fields.linear_mode.i_mode
        delta_k_adim = params.init_fields.linear_mode.delta_k_adim
        amplitude = params.init_fields.linear_mode.amplitude * params.N**2

        # Define linear mode to put energy
        linear_mode = np.zeros(oper.shapeK, dtype="complex")

        # Define grid of indices (sequential or parallel)
        if mpi.nb_proc > 1:
            ikx_mode = np.arange(
                oper.kx_loc[0] / oper.deltakx,
                (oper.kx_loc[-1] + oper.deltakx) / oper.deltakx,
                dtype=int,
            )
            iky_mode = np.arange(0, oper.shapeK[1], dtype=int)

            iKY, iKX = np.meshgrid(iky_mode, ikx_mode)

        else:
            ikx_mode = np.arange(0, oper.shapeK[0])
            iky_mode = np.arange(0, oper.shapeK[1])

            iKX, iKY = np.meshgrid(ikx_mode, iky_mode)

        # Condition to put energy
        cond_x = np.logical_and(
            iKX >= i_mode[0], iKX <= i_mode[0] + delta_k_adim - 1
        )

        cond_y = np.logical_and(
            iKY >= i_mode[1], iKY <= i_mode[1] + delta_k_adim - 1
        )

        COND = np.logical_and(cond_x, cond_y)

        indices_cond = np.argwhere(COND == True)

        for index in indices_cond:
            linear_mode[tuple(index)] = 1.0 + 0j
            # linear_mode[tuple(index)] = random.random() + 0j

        return amplitude * linear_mode


class InitFieldsNS2DStrat(InitFieldsBase):
    """Initialize the state for the solver ns2d.strat."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""
        InitFieldsBase._complete_info_solver(
            info_solver,
            classes=[
                InitFieldsNoiseStrat,
                InitFieldsJetStrat,
                InitFieldsDipoleStrat,
                InitFieldsLinearMode,
            ],
        )
