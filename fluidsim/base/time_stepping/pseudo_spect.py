"""Time stepping (:mod:`fluidsim.base.time_stepping.pseudo_spect`)
========================================================================

Provides:

.. autoclass:: TimeSteppingPseudoSpectral
   :members:
   :private-members:

.. todo::

  It would be interesting to implement phase-shifting timestepping schemes as:

  - RK2 + phase-shifting

  - Adams-Bashforth (leapfrog) + phase-shifting

  For a theoretical presentation of phase-shifting see
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19810022965.pdf.

"""

import numpy as np

from transonic import Transonic, Type, NDim, Array

from .base import TimeSteppingBase

ts = Transonic()

N = NDim(2, 3, 4)
A = Array[np.complex128, N, "C"]

T = Type(np.float64, np.complex128)
A1 = Array[T, N, "C"]
A2 = Array[T, N - 1, "C"]

uniform = np.random.default_rng().uniform


class ExactLinearCoefs:
    """Handle the computation of the exact coefficient for the RK4."""

    def __init__(self, time_stepping):
        self.time_stepping = time_stepping
        sim = time_stepping.sim
        self.shapeK_loc = sim.oper.shapeK_loc
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
        exact = self.exact
        exact2 = self.exact2

        if ts.is_transpiled:
            ts.use_block("exact_lin_compute")
        else:
            # transonic block (
            #     A1 f_lin, exact, exact2;
            #     float dt
            # )
            # transonic block (
            #     A2 f_lin, exact, exact2;
            #     float dt
            # )
            exact[:] = np.exp(-dt * f_lin)
            exact2[:] = np.exp(-dt / 2 * f_lin)
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


class TimeSteppingPseudoSpectral(TimeSteppingBase):
    """Time stepping class for pseudo-spectral solvers.

    """

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        TimeSteppingBase._complete_params_with_default(params)
        params.time_stepping.USE_CFL = True

    def __init__(self, sim):
        super().__init__(sim)
        self.init_from_params()

    def init_from_params(self):
        self._init_freq_lin()
        self._init_compute_time_step()
        self._init_exact_linear_coef()
        self._init_time_scheme()

    def _init_freq_lin(self):
        f_d, f_d_hypo = self.sim.compute_freq_diss()
        freq_dissip = f_d + f_d_hypo

        if hasattr(self.sim, "compute_freq_complex"):
            freq_complex = self._compute_freq_complex()
            self.freq_lin = freq_dissip + freq_complex
            freq_max = freq_complex.imag.max()
            self.deltat_max = 0.78 * np.pi / freq_max
        else:
            self.freq_lin = freq_dissip

    def _init_time_scheme(self):

        type_time_scheme = self.params.time_stepping.type_time_scheme

        if type_time_scheme.startswith("RK"):
            self._state_spect_tmp = np.empty_like(self.sim.state.state_spect)

        if type_time_scheme == "Euler":
            time_step_RK = self._time_step_Euler
        elif type_time_scheme == "Euler_phaseshift":
            time_step_RK = self._time_step_Euler_phaseshift
        elif type_time_scheme == "Euler_phaseshift_random":
            time_step_RK = self._time_step_Euler_phaseshift_random
        elif type_time_scheme == "RK2":
            time_step_RK = self._time_step_RK2
        elif type_time_scheme == "RK2_trapezoid":
            time_step_RK = self._time_step_RK2_trapezoid
        elif type_time_scheme == "RK2_phaseshift":
            time_step_RK = self._time_step_RK2_phaseshift
        elif type_time_scheme == "RK2_phaseshift_random":
            time_step_RK = self._time_step_RK2_phaseshift_random
        elif type_time_scheme == "RK2_phaseshift_exact":
            time_step_RK = self._time_step_RK2_phaseshift_exact
        elif type_time_scheme == "RK4":
            self._state_spect_tmp1 = np.empty_like(self.sim.state.state_spect)
            time_step_RK = self._time_step_RK4
        else:
            raise ValueError(f'Problem name time_scheme ("{type_time_scheme}")')

        self._time_step_RK = time_step_RK

    def _compute_freq_complex(self):
        state_spect = self.sim.state.state_spect
        freq_complex = np.empty_like(state_spect)
        for ik, key in enumerate(state_spect.keys):
            freq_complex[ik] = self.sim.compute_freq_complex(key)
        return freq_complex

    def _init_exact_linear_coef(self):
        self.exact_linear_coefs = ExactLinearCoefs(self)

    def one_time_step_computation(self):
        """One time step."""
        # WARNING: if the function _time_step_RK comes from an extension, its
        # execution time seems to be attributed to the function
        # one_time_step_computation by cProfile
        self._time_step_RK()
        self.sim.oper.dealiasing(self.sim.state.state_spect)
        self.sim.state.statephys_from_statespect()
        # np.isnan(np.sum seems to be really fast
        if np.isnan(np.sum(self.sim.state.state_spect[0])):
            raise ValueError(f"nan at it = {self.it}, t = {self.t:.4f}")

    def _time_step_Euler(self):
        r"""Forward Euler method.

        Notes
        -----

        .. |p| mathmacro:: \partial
        .. |dt| mathmacro:: \mathop{dt}
        .. |dx| mathmacro:: \mathop{dx}

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        The forward Euler method computes an approximation of the
        solution after a time increment :math:`\dt`. We denote the
        initial time :math:`t = 0`.

        Euler approximation :

          .. math:: \p_t \log S = \sigma + \frac{N_0}{S_0},

        Integrating from :math:`t` to :math:`t+\dt`, it gives:

          .. math:: S_{\dt} = (S_0 + N_0 \dt) e^{\sigma \dt}.

        """
        dt = self.deltat
        diss = self.exact_linear_coefs.get_updated_coefs()[0]

        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        tendencies_0 = compute_tendencies()

        state_spect[:] = (state_spect + dt * tendencies_0) * diss

    def _get_phase_shift(self):
        """Compute the phase shift term."""
        phase = 0.5 * self.sim.oper.deltax * self.sim.oper.kx
        return np.exp(1j * phase)

    def _get_phase_shift_random(self):
        """Compute two random phase shift terms."""
        alpha = uniform(-1, 1)
        if alpha < 0:
            beta = alpha + 0.5
        else:
            beta = alpha - 0.5

        phase_shift_alpha = np.exp(
            1j * alpha * self.sim.oper.deltax * self.sim.oper.kx
        )
        phase_shift_beta = np.exp(
            1j * beta * self.sim.oper.deltax * self.sim.oper.kx
        )

        return phase_shift_alpha, phase_shift_beta

    def _time_step_Euler_phaseshift(self):
        r"""Forward Euler method, dealiasing with phase-shifting.

        Notes
        -----

        WIP: only for 1D!

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        The forward Euler method computes an approximation of the
        solution after a time increment :math:`\dt`. We denote the
        initial time :math:`t = 0`.

        - Euler approximation:

          .. math:: \p_t \log S = \sigma + \frac{N_\mathrm{dealias}}{S_0},

          Integrating from :math:`t` to :math:`t+\dt`, it gives:

          .. math:: S_{\dt} = (S_0 + N_\mathrm{dealias} \dt) e^{\sigma \dt}.

          where the dealiased non-linear term is :math:`N_\mathrm{dealias} =
          (N_0 + \tilde N_0)/2` and the phase-shifted nonlinear term
          :math:`\tilde N_0` is given by

          .. math::
            \tilde N_0 = e^{-\frac{1}{2}k\dx}N\left(e^{\frac{1}{2}k\dx}S_0\right).

        """
        dt = self.deltat
        diss = self.exact_linear_coefs.get_updated_coefs()[0]

        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        # regular tendencies
        tendencies_0 = compute_tendencies()
        # shifted tendencies
        phase_shift = self._get_phase_shift()
        tendencies_shifted = (
            compute_tendencies(phase_shift * state_spect) / phase_shift
        )
        # dealiased tendencies
        tendencies_dealiased = 0.5 * (tendencies_0 + tendencies_shifted)

        state_spect[:] = (state_spect + dt * tendencies_dealiased) * diss

    def _time_step_Euler_phaseshift_random(self):
        r"""Forward Euler method, dealiasing with phase-shifting.

        Notes
        -----

        WIP: only for 1D!

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        The forward Euler method computes an approximation of the
        solution after a time increment :math:`\dt`. We denote the
        initial time :math:`t = 0`.

        - Euler approximation:

          .. math:: \p_t \log S = \sigma + \frac{N_\mathrm{dealias}}{S_0},

          Integrating from :math:`t` to :math:`t+\dt`, it gives:

          .. math:: S_{\dt} = (S_0 + N_\mathrm{dealias} \dt) e^{\sigma \dt}.

          where the dealiased non-linear term :math:`N_\mathrm{dealias} =
          (\tilde N_{0\alpha} + \tilde N_{0\beta})/2` is computed as the
          average of two terms shifted with dependant phases
          :math:`\phi_\alpha = \alpha \dx k` and :math:`\phi_\beta = \beta \dx
          k` with :math:`\alpha` taken randomly between -1 and 1 and
          :math:`|\alpha - \beta| = 0.5`.

        """
        dt = self.deltat
        diss = self.exact_linear_coefs.get_updated_coefs()[0]

        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        # shifted tendencies
        phase_shift_alpha, phase_shift_beta = self._get_phase_shift_random()
        tendencies_alpha = (
            compute_tendencies(phase_shift_alpha * state_spect)
            / phase_shift_alpha
        )
        tendencies_beta = (
            compute_tendencies(phase_shift_beta * state_spect) / phase_shift_beta
        )
        # dealiased tendencies
        tendencies_dealiased = 0.5 * (tendencies_alpha + tendencies_beta)

        state_spect[:] = (state_spect + dt * tendencies_dealiased) * diss

    def _time_step_RK2(self):
        r"""Runge-Kutta 2 method.

        Notes
        -----

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        The Runge-Kutta 2 method computes an approximation of the
        solution after a time increment :math:`\dt`. We denote the
        initial time :math:`t = 0`.

        - Approximation 1:

          .. math:: \p_t \log S = \sigma + \frac{N_0}{S_0},

          Integrating from :math:`t` to :math:`t+\dt/2`, it gives:

          .. math:: S_1 = (S_0 + \frac{\dt}{2} N_0) e^{\sigma \frac{\dt}{2}}.

        - Approximation 2:

          .. math::
             \p_t \log S = \sigma + \frac{N_1}{S_1},

          Integrating from :math:`t` to :math:`t+\dt` and retaining
          only the terms in :math:`(N\dt/S)^1` gives:

          .. math::
             S_2 = S_0 e^{\sigma \dt} + \dt N_1 e^{\sigma \frac{\dt}{2}}.

        """
        dt = self.deltat
        diss, diss2 = self.exact_linear_coefs.get_updated_coefs()

        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        tendencies_0 = compute_tendencies()

        state_spect_12 = self._state_spect_tmp

        if ts.is_transpiled:
            ts.use_block("rk2_step0")
        else:
            # transonic block (
            #     A state_spect_12, state_spect, tendencies_0;
            #     A1 diss2;
            #     float dt
            # )
            # transonic block (
            #     A state_spect_12, state_spect, tendencies_0;
            #     A2 diss2;
            #     float dt
            # )

            state_spect_12[:] = (state_spect + dt / 2 * tendencies_0) * diss2

        tendencies_12 = compute_tendencies(state_spect_12, old=tendencies_0)

        if ts.is_transpiled:
            ts.use_block("rk2_step1")
        else:
            # transonic block (
            #     A state_spect, tendencies_12;
            #     A1 diss, diss2;
            #     float dt
            # )

            # transonic block (
            #     A state_spect, tendencies_12;
            #     A2 diss, diss2;
            #     float dt
            # )

            state_spect[:] = state_spect * diss + dt * diss2 * tendencies_12

    def _time_step_RK2_trapezoid(self):
        r"""Runge-Kutta 2 method with trapezoidal rule (Heun's method).

        Notes
        -----

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        Heun's method computes an approximation of the
        solution after a time increment :math:`\dt`. We denote the
        initial time :math:`t = 0`.

        - Approximation 1:

          .. math:: \p_t \log S = \sigma + \frac{N(S_0)}{S_0},

          Integrating from :math:`t` to :math:`t+\dt`, it gives:

          .. math:: S_1 = (S_0 + N_0 \dt) e^{\sigma \dt}.

        - Approximation 2:

          .. math::
             \p_t \log S = \sigma + \frac{1}{2}\left(
                 \frac{N_0}{S_0} + \frac{N_1}{S_1}\right),

          Integrating from :math:`t` to :math:`t+\dt` and retaining
          only the terms in :math:`(N\dt/S)^1` gives:

          .. math::
             S_2 = S_0 e^{\sigma \dt} + \frac{\dt}{2} (N_0 e^{\sigma \dt} + N_1).

        """
        dt = self.deltat
        diss, diss2 = self.exact_linear_coefs.get_updated_coefs()

        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        tendencies_0 = compute_tendencies()

        state_spect_1 = self._state_spect_tmp

        state_spect_1[:] = (state_spect + dt * tendencies_0) * diss

        tendencies_1 = compute_tendencies(state_spect_1)

        state_spect[:] = (
            state_spect + dt / 2 * tendencies_0
        ) * diss + dt / 2 * tendencies_1

    def _time_step_RK2_phaseshift(self):
        r"""Runge-Kutta 2 method with phase-shifting.

        Notes
        -----

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        Heun's method computes an approximation of the
        solution after a time increment :math:`\dt`. We denote the
        initial time :math:`t = 0`.

        - Approximation 1:

          .. math:: \p_t \log S = \sigma + \frac{N_0}{S_0},

          Integrating from :math:`t` to :math:`t+\dt`, it gives:

          .. math:: S_1 = (S_0 + N_0 \dt) e^{\sigma \dt}.

        - Approximation 2:

          .. math::
             \p_t \log S = \sigma + \frac{N_d}{S_0 e^{\sigma \frac{\dt}{2}}},


          where the dealiased non-linear term is :math:`N_\mathrm{dealias} =
          (N_0 + \tilde N_1)/2` and the phase-shifted nonlinear term
          :math:`\tilde N_1` is given by

          .. math::
            \tilde N_1 = e^{-\frac{1}{2}k\dx}N\left(e^{\frac{1}{2}k\dx}S_1\right).

          Integrating from :math:`t` to :math:`t+\dt` and retaining
          only the terms in :math:`(N\dt/S)^1` gives:

          .. math::
             S_2 = S_0 e^{\sigma \dt} + \dt N_d e^{\sigma \frac{\dt}{2}}.

        """
        dt = self.deltat
        diss, diss2 = self.exact_linear_coefs.get_updated_coefs()

        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        tendencies_0 = compute_tendencies()

        state_spect_1 = self._state_spect_tmp

        state_spect_1[:] = (state_spect + dt * tendencies_0) * diss

        phase_shift = self._get_phase_shift()
        tendencies_1 = (
            compute_tendencies(phase_shift * state_spect_1) / phase_shift
        )

        tendencies_d = (tendencies_0 + tendencies_1) / 2
        state_spect[:] = state_spect * diss + dt * tendencies_d * diss2

    def _time_step_RK2_phaseshift_random(self):
        r"""Runge-Kutta 2 method with phase-shifting (random).

        Notes
        -----

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        Heun's method computes an approximation of the
        solution after a time increment :math:`\dt`. We denote the
        initial time :math:`t = 0`.

        - Approximation 1:

          .. math:: \p_t \log S = \sigma + \frac{N_0}{S_0},

          Integrating from :math:`t` to :math:`t+\dt`, it gives:

          .. math:: S_1 = (S_0 + N_0 \dt) e^{\sigma \dt}.

        - Approximation 2:

          .. math::
             \p_t \log S = \sigma + \frac{N_d}{S_0 e^{\sigma \frac{\dt}{2}}},

          where the dealiased non-linear term is :math:`N_d =
          (\tilde N_{0\alpha} + \tilde N_{1\beta})/2`.

          Integrating from :math:`t` to :math:`t+\dt` and retaining
          only the terms in :math:`(N\dt/S)^1` gives:

          .. math::
             S_2 = S_0 e^{\sigma \dt} + \dt N_d e^{\sigma \frac{\dt}{2}}.

        """
        dt = self.deltat
        diss, diss2 = self.exact_linear_coefs.get_updated_coefs()

        phase_shift_alpha, phase_shift_beta = self._get_phase_shift_random()

        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        tendencies_0 = (
            compute_tendencies(phase_shift_alpha * state_spect)
            / phase_shift_alpha
        )

        state_spect_1 = self._state_spect_tmp

        state_spect_1[:] = (state_spect + dt * tendencies_0) * diss

        tendencies_1 = (
            compute_tendencies(phase_shift_beta * state_spect_1)
            / phase_shift_beta
        )

        tendencies_d = (tendencies_0 + tendencies_1) / 2
        state_spect[:] = state_spect * diss + dt * tendencies_d * diss2

    def _time_step_RK2_phaseshift_exact(self):
        r"""Runge-Kutta 2 method with phase-shifting for exact dealiasing.

        Notes
        -----

        It requires 4 evaluations of the nonlinear terms (as RK4).

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        Heun's method computes an approximation of the
        solution after a time increment :math:`\dt`. We denote the
        initial time :math:`t = 0`.

        - Approximation 1:

          .. math:: \p_t \log S = \sigma + \frac{N_{d0}}{S_0},

          Integrating from :math:`t` to :math:`t+\dt`, it gives:

          .. math:: S_1 = (S_0 + N_{d0} \dt) e^{\sigma \dt}.

        - Approximation 2:

          .. math::
             \p_t \log S = \sigma + \frac{N_d}{S_0 e^{\sigma \frac{\dt}{2}}},

          where the dealiased non-linear term is :math:`N_d =
          (N_{d0} +  N_{d1})/2`.

          Integrating from :math:`t` to :math:`t+\dt` and retaining
          only the terms in :math:`(N\dt/S)^1` gives:

          .. math::
             S_2 = S_0 e^{\sigma \dt} + \dt N_d e^{\sigma \frac{\dt}{2}}.

        """
        dt = self.deltat
        diss, diss2 = self.exact_linear_coefs.get_updated_coefs()

        phase_shift = self._get_phase_shift()
        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        tendencies_d0 = (
            compute_tendencies(state_spect)
            + compute_tendencies(phase_shift * state_spect) / phase_shift
        ) / 2

        state_spect_1 = self._state_spect_tmp

        state_spect_1[:] = (state_spect + dt * tendencies_d0) * diss

        tendencies_d1 = (
            compute_tendencies(state_spect_1)
            + compute_tendencies(phase_shift * state_spect_1) / phase_shift
        ) / 2

        tendencies_d = (tendencies_d0 + tendencies_d1) / 2
        state_spect[:] = state_spect * diss + dt * tendencies_d * diss2

    def _time_step_RK4(self):
        r"""Runge-Kutta 4 method.

        Notes
        -----

        .. |SA1dt2| mathmacro:: S_{A1 \mathop{dt}/2}

        We consider an equation of the form

        .. math:: \p_t S = \sigma S + N(S),

        The Runge-Kutta 4 method computes an approximation of the
        solution after a time increment :math:`\dt`. We denote the
        initial time as :math:`t = 0`. This time scheme uses 4
        approximations. Only the terms in :math:`\dt^1` are retained.

        - Approximation 1:

          .. math:: \p_t \log S = \sigma + \frac{N(S_0)}{S_0},

          Integrating from :math:`t` to :math:`t+\dt/2` gives:

          .. math:: \SA1dt2 = (S_0 + N_0 \dt/2) e^{\sigma \frac{\dt}{2}}.

          Integrating from :math:`t` to :math:`t+\dt` gives:

          .. math:: S_{A1\dt} = (S_0 + N_0 \dt) e^{\sigma \dt}.

        - Approximation 2:

          .. math::
             \p_t \log S = \sigma
             + \frac{N(\SA1dt2)}{ \SA1dt2 },

          Integrating from :math:`t` to :math:`t+\dt/2` gives:

          .. |SA2dt2| mathmacro:: S_{A2 \mathop{dt}/2}

          .. math::
             \SA2dt2 = S_0 e^{\sigma \frac{\dt}{2}}
             + N(\SA1dt2) \frac{\dt}{2}.

          Integrating from :math:`t` to :math:`t+\dt` gives:

          .. math::
             S_{A2\dt} = S_0 e^{\sigma \dt}
             + N(\SA1dt2) e^{\sigma \frac{\dt}{2}} \dt.


        - Approximation 3:

          .. math::
             \p_t \log S = \sigma
             + \frac{N(\SA2dt2)}{ \SA2dt2 },

          Integrating from :math:`t` to :math:`t+\dt` gives:

          .. math::
             S_{A3\dt} = S_0 e^{\sigma \dt}
             + N(\SA2dt2) e^{\sigma \frac{\dt}{2}} \dt.

        - Approximation 4:

          .. math::
             \p_t \log S = \sigma
             + \frac{N(S_{A3\dt})}{ S_{A3\dt} },

          Integrating from :math:`t` to :math:`t+\dt` gives:

          .. math::
             S_{A4\dt} = S_0 e^{\sigma \dt} + N(S_{A3\dt}) \dt.

        The final result is a pondered average of the results of 4
        approximations for the time :math:`t+\dt`:

          .. math::
             \frac{1}{3} \left[
             \frac{1}{2} S_{A1\dt}
             + S_{A2\dt} + S_{A3\dt}
             + \frac{1}{2} S_{A4\dt}
             \right],

        which is equal to:

          .. math::
             S_0 e^{\sigma \dt}
             + \frac{\dt}{3} \left[
             \frac{1}{2} N(S_0) e^{\sigma \dt}
             + N(\SA1dt2) e^{\sigma \frac{\dt}{2}}
             + N(\SA2dt2) e^{\sigma \frac{\dt}{2}}
             + \frac{1}{2} N(S_{A3\dt})\right].

        """
        dt = self.deltat
        diss, diss2 = self.exact_linear_coefs.get_updated_coefs()

        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        tendencies_0 = compute_tendencies()
        state_spect_tmp = self._state_spect_tmp
        state_spect_tmp1 = self._state_spect_tmp1
        state_spect_12_approx1 = state_spect_tmp1

        if ts.is_transpiled:
            ts.use_block("rk4_step0")
        else:
            # based on approximation 0
            # transonic block (
            #     A state_spect, state_spect_tmp,
            #       tendencies_0, state_spect_12_approx1;
            #     A1 diss, diss2;
            #     float dt
            # )

            # transonic block (
            #     A state_spect, state_spect_tmp,
            #       tendencies_0, state_spect_12_approx1;
            #     A2 diss, diss2;
            #     float dt
            # )

            state_spect_tmp[:] = (state_spect + dt / 6 * tendencies_0) * diss
            state_spect_12_approx1[:] = (
                state_spect + dt / 2 * tendencies_0
            ) * diss2

        tendencies_1 = compute_tendencies(
            state_spect_12_approx1, old=tendencies_0
        )
        del state_spect_12_approx1

        state_spect_12_approx2 = state_spect_tmp1

        if ts.is_transpiled:
            ts.use_block("rk4_step1")
        else:
            # based on approximation 1
            # transonic block (
            #     A state_spect, state_spect_tmp,
            #       state_spect_12_approx2, tendencies_1;
            #     A1 diss2;
            #     float dt
            # )

            # transonic block (
            #     A state_spect, state_spect_tmp,
            #       state_spect_12_approx2, tendencies_1;
            #     A2 diss2;
            #     float dt
            # )

            state_spect_tmp[:] += dt / 3 * diss2 * tendencies_1
            state_spect_12_approx2[:] = (
                state_spect * diss2 + dt / 2 * tendencies_1
            )

        tendencies_2 = compute_tendencies(
            state_spect_12_approx2, old=tendencies_1
        )
        del state_spect_12_approx2

        state_spect_1_approx = state_spect_tmp1

        if ts.is_transpiled:
            ts.use_block("rk4_step2")
        else:
            # based on approximation 2
            # transonic block (
            #     A state_spect, state_spect_tmp,
            #       state_spect_1_approx, tendencies_2;
            #     A1 diss, diss2;
            #     float dt
            # )

            # transonic block (
            #     A state_spect, state_spect_tmp,
            #       state_spect_1_approx, tendencies_2;
            #     A2 diss, diss2;
            #     float dt
            # )

            state_spect_tmp[:] += dt / 3 * diss2 * tendencies_2
            state_spect_1_approx[:] = (
                state_spect * diss + dt * diss2 * tendencies_2
            )

        tendencies_3 = compute_tendencies(state_spect_1_approx, old=tendencies_2)
        del state_spect_1_approx

        if ts.is_transpiled:
            ts.use_block("rk4_step3")
        else:
            # result using the 4 approximations
            # transonic block (
            #     A state_spect, state_spect_tmp, tendencies_3;
            #     float dt
            # )
            state_spect[:] = state_spect_tmp + dt / 6 * tendencies_3
