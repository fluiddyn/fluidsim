"""Time stepping (:mod:`fluidsim.solvers.sw1l.time_stepping`)
================================================================

.. currentmodule:: fluidsim.solvers.sw1l.time_stepping

Provides:

.. autoclass:: TimeSteppingSW1L
   :members:
   :private-members:

"""

import numpy as np
from fluidsim.base.time_stepping.pseudo_spect_cy import TimeSteppingPseudoSpectral


class TimeSteppingSW1L(TimeSteppingPseudoSpectral):

    @staticmethod
    def _complete_params_with_default(params):
        super(TimeSteppingSW1L, TimeSteppingSW1L)._complete_params_with_default(params)
        params.time_stepping._set_attribs({'forcing_time_scheme': 'RK'})

    def one_time_step_computation(self):
        """One time step for SW1L solver and its sub-solvers"""

        self._time_step_RK()
        if self.params.FORCING and self.params.time_stepping.forcing_time_scheme == 'EULER':
            self._time_step_euler_forcing()

        self.sim.oper.dealiasing(self.sim.state.state_fft)
        self.sim.state.statephys_from_statefft()
        if np.isnan(np.min(self.sim.state.state_fft[0])):
            raise ValueError(
                'nan at it = {0}, t = {1:.4f}'.format(self.it, self.t))

    def _time_step_euler_forcing(self):
        """
        Separate scheme for forcing to avoid miscalculations of forcing rate
        possibly arising out of multi stage time stepping.
        """
        
        dt = self.deltat
        state_fft = self.sim.state.state_fft
        forcing_fft = self.sim.forcing.get_forcing()
        state_fft += forcing_fft * dt
