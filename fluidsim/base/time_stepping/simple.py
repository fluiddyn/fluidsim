"""Time stepping (:mod:`fluidsim.base.time_stepping.simple`)
============================================================


Provides:

.. autoclass:: TimeSteppingSimple
   :members:
   :private-members:

"""

import numpy as np

from .base import TimeSteppingBase


class ExactLinearCoefs:
    """Handle the computation of the exact coefficient for the RK4."""

    def __init__(self, time_stepping):
        self.time_stepping = time_stepping
        sim = time_stepping.sim
        self.freq_lin = time_stepping.freq_lin

        self.exact = np.empty_like(self.freq_lin)
        self.exact2 = np.empty_like(self.freq_lin)

        if sim.params.time_stepping.USE_CFL:
            self.get_updated_coefs = self.get_updated_coefs_CLF
            self.dt_old = 0.0
        else:
            self.compute(time_stepping.deltat)
            self.get_updated_coefs = self.get_coefs

    def compute(self, dt):
        """Compute the exact coefficients."""
        f_lin = self.freq_lin
        self.exact = np.exp(-dt * f_lin)
        self.exact2 = np.exp(-dt / 2 * f_lin)
        self.dt_old = dt

    def get_updated_coefs_CLF(self):
        """Get the exact coefficient updated if needed."""
        dt = self.time_stepping.deltat
        if self.dt_old != dt:
            self.compute(dt)
        return self.exact, self.exact2

    def get_coefs(self):
        """Get the exact coefficients as stored."""
        return self.exact, self.exact2


class TimeSteppingSimple(TimeSteppingBase):
    """Time stepping class for pseudo-spectral solvers."""

    def __init__(self, sim):
        super().__init__(sim)

        self._init_freq_lin()
        self._init_compute_time_step()
        self._init_exact_linear_coef()
        self._init_time_scheme()

    def _init_freq_lin(self):
        self.freq_lin = 0.0

    def _init_exact_linear_coef(self):
        self.exact_linear_coefs = ExactLinearCoefs(self)

    def one_time_step_computation(self):
        """One time step"""
        # warning: if the function _time_step_RK comes from an extension, its
        # execution time seems to be attributed to the function
        # one_time_step_computation by cProfile
        self._time_step_RK()
        if np.isnan(np.sum(self.sim.state.state_phys[0])):
            raise ValueError(f"nan at it = {self.it}, t = {self.t:.4f}")

    def _time_step_RK2(self):
        r"""Advance in time with the Runge-Kutta 2 method.

        .. _rk2timescheme:

        Notes
        -----

        .. |p| mathmacro:: \partial

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        The Runge-Kutta 2 method computes an approximation of the
        solution after a time increment :math:`dt`. We denote the
        initial time :math:`t = 0`.

        - Approximation 1:

          .. math:: \p_t \log S = \sigma + \frac{N(S_0)}{S_0},

          Integrating from :math:`t` to :math:`t+dt/2`, it gives:

          .. |SA1halfdt| mathmacro:: S_{A1dt/2}

          .. math:: \SA1halfdt = (S_0 + N_0 dt/2) e^{\frac{\sigma dt}{2}}.


        - Approximation 2:

          .. math::
             \p_t \log S = \sigma
             + \frac{N(\SA1halfdt)}{ \SA1halfdt },

          Integrating from :math:`t` to :math:`t+dt` and retaining
          only the terms in :math:`dt^1` gives:

          .. math::
             S_{dtA2} = S_0 e^{\sigma dt}
             + N(\SA1halfdt) dt e^{\frac{\sigma dt}{2}}.

        """
        dt = self.deltat
        diss, diss2 = self.exact_linear_coefs.get_updated_coefs()

        tendencies_nonlin = self.sim.tendencies_nonlin
        state_phys = self.sim.state.state_phys

        tendencies_n = tendencies_nonlin()
        state_phys_n12 = (state_phys + dt / 2 * tendencies_n) * diss2
        tendencies_phys_n12 = tendencies_nonlin(state_phys_n12)
        self.sim.state.state_phys = (
            state_phys * diss + dt * diss2 * tendencies_phys_n12
        )

    def _time_step_RK4(self):
        r"""Advance in time with the Runge-Kutta 4 method.

        .. _rk4timescheme:

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        The Runge-Kutta 4 method computes an approximation of the
        solution after a time increment :math:`dt`. We denote the
        initial time as :math:`t = 0`. This time scheme uses 4
        approximations. Only the terms in :math:`dt^1` are retained.

        - Approximation 1:

          .. math:: \p_t \log S = \sigma + \frac{N(S_0)}{S_0},

          Integrating from :math:`t` to :math:`t+dt/2` gives:

          .. math:: \SA1halfdt = (S_0 + N_0 dt/2) e^{\sigma \frac{dt}{2}}.

          Integrating from :math:`t` to :math:`t+dt` gives:

          .. math:: S_{A1dt} = (S_0 + N_0 dt) e^{\sigma dt}.


        - Approximation 2:

          .. math::
             \p_t \log S = \sigma
             + \frac{N(\SA1halfdt)}{ \SA1halfdt },

          Integrating from :math:`t` to :math:`t+dt/2` gives:

          .. |SA2halfdt| mathmacro:: S_{A2 dt/2}

          .. math::
             \SA2halfdt = S_0 e^{\sigma \frac{dt}{2}}
             + N(\SA1halfdt) \frac{dt}{2}.

          Integrating from :math:`t` to :math:`t+dt` gives:

          .. math::
             S_{A2dt} = S_0 e^{\sigma dt}
             + N(\SA1halfdt) e^{\sigma \frac{dt}{2}} dt.


        - Approximation 3:

          .. math::
             \p_t \log S = \sigma
             + \frac{N(\SA2halfdt)}{ \SA2halfdt },

          Integrating from :math:`t` to :math:`t+dt` gives:

          .. math::
             S_{A3dt} = S_0 e^{\sigma dt}
             + N(\SA2halfdt) e^{\sigma \frac{dt}{2}} dt.

        - Approximation 4:

          .. math::
             \p_t \log S = \sigma
             + \frac{N(S_{A3dt})}{ S_{A3dt} },

          Integrating from :math:`t` to :math:`t+dt` gives:

          .. math::
             S_{A4dt} = S_0 e^{\sigma dt} + N(S_{A3dt}) dt.


        The final result is a pondered average of the results of 4
        approximations for the time :math:`t+dt`:

          .. math::
             \frac{1}{3} \left[
             \frac{1}{2} S_{A1dt}
             + S_{A2dt} + S_{A3dt}
             + \frac{1}{2} S_{A4dt}
             \right],

        which is equal to:

          .. math::
             S_0 e^{\sigma dt}
             + \frac{dt}{3} \left[
             \frac{1}{2} N(S_0) e^{\sigma dt}
             + N(\SA1halfdt) e^{\sigma \frac{dt}{2}}
             + N(\SA2halfdt) e^{\sigma \frac{dt}{2}}
             + \frac{1}{2} N(S_{A3dt})\right].

        """

        dt = self.deltat
        diss, diss2 = self.exact_linear_coefs.get_updated_coefs()

        tendencies_nonlin = self.sim.tendencies_nonlin
        state_phys = self.sim.state.state_phys

        tendencies_0 = tendencies_nonlin()

        # based on approximation 1
        state_phys_temp = (state_phys + dt / 6 * tendencies_0) * diss
        state_phys_np12_approx1 = (state_phys + dt / 2 * tendencies_0) * diss2

        del tendencies_0
        tendencies_1 = tendencies_nonlin(state_phys_np12_approx1)
        del state_phys_np12_approx1

        # based on approximation 2
        state_phys_temp += dt / 3 * diss2 * tendencies_1
        state_phys_np12_approx2 = state_phys * diss2 + dt / 2 * tendencies_1

        del tendencies_1
        tendencies_2 = tendencies_nonlin(state_phys_np12_approx2)
        del state_phys_np12_approx2

        # based on approximation 3
        state_phys_temp += dt / 3 * diss2 * tendencies_2
        state_phys_np1_approx = state_phys * diss + dt * diss2 * tendencies_2

        del tendencies_2
        tendencies_3 = tendencies_nonlin(state_phys_np1_approx)
        del state_phys_np1_approx

        # result using the 4 approximations
        self.sim.state.state_phys = state_phys_temp + dt / 6 * tendencies_3
