"""Time stepping (:mod:`fluidsim.base.time_stepping.pseudo_spect`)
========================================================================

Provides:

.. autoclass:: TimeSteppingPseudoSpectral
   :members:
   :private-members:

.. todo::

  It would be interesting to also implement the Adams-Bashforth (leapfrog)
  scheme with phase-shifting. It is very close to
  :func:`_time_step_RK2_phaseshift` with 2 evaluations of the non-linear terms
  per time step (but with 2 symmetrical and equivalent steps).

.. note::

  For a theoretical presentation of phase-shifting, see the book Numerical
  Experiments in Homogeneous Turbulence (Robert S. Rogallo,
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19810022965.pdf).

"""
from random import randint

import numpy as np

from transonic import Transonic, Type, NDim, Array, boost, Union

from .base import TimeSteppingBase

ts = Transonic()

N = NDim(2, 3, 4)
A = Array[np.complex128, N, "C"]
Am1 = Array[np.complex128, N - 1, "C"]

N123 = NDim(1, 2, 3)
A123c = Array[np.complex128, N123, "C"]
A123f = Array[np.float64, N123, "C"]

T = Type(np.float64, np.complex128)
A1 = Array[T, N, "C"]
A2 = Array[T, N - 1, "C"]
ArrayDiss = Union[A1, A2]

uniform = np.random.default_rng().uniform


@boost
def step_Euler(
    state_spect: A, dt: float, tendencies: A, diss: ArrayDiss, output: A
):
    output[:] = (state_spect + dt * tendencies) * diss
    return output


@boost
def step_Euler_inplace(state_spect: A, dt: float, tendencies: A, diss: ArrayDiss):
    step_Euler(state_spect, dt, tendencies, diss, state_spect)


@boost
def step_like_RK2(
    state_spect: A, dt: float, tendencies: A, diss: ArrayDiss, diss2: ArrayDiss
):
    state_spect[:] = state_spect * diss + dt * diss2 * tendencies


@boost
def mean_with_phaseshift(
    tendencies_0: A, tendencies_1_shift: A, phaseshift: Am1, output: A
):
    output[:] = 0.5 * (tendencies_0 + tendencies_1_shift / phaseshift)
    return output


@boost
def mul(phaseshift: Am1, state_spect: A, output: A):
    output[:] = phaseshift * state_spect
    return output


@boost
def div_inplace(arr: A, phaseshift: Am1):
    arr /= phaseshift
    return arr


@boost
def compute_phaseshift_terms(
    phase_alpha: A123f,
    phase_beta: A123f,
    phaseshift_alpha: A123c,
    phaseshift_beta: A123c,
):
    phaseshift_alpha[:] = np.exp(1j * phase_alpha)
    phaseshift_beta[:] = np.exp(1j * phase_beta)
    return phaseshift_alpha, phaseshift_beta


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
    """Time stepping class for pseudo-spectral solvers."""

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""
        TimeSteppingBase._complete_params_with_default(params)
        params.time_stepping.USE_CFL = True

        params.time_stepping._set_child(
            "phaseshift_random",
            attribs=dict(nb_pairs=1, nb_steps_compute_new_pair=None),
        )

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

        if type_time_scheme.endswith("_random"):
            self._init_phaseshift_random()
            if not hasattr(self.sim.oper, "get_phases_random"):
                raise NotImplementedError

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
        step_Euler_inplace(state_spect, dt, tendencies_0, diss)

    def _get_phaseshift(self):
        """Compute the phase shift term."""
        if hasattr(self, "_phaseshift") and self._phaseshift is not None:
            return self._phaseshift
        oper = self.sim.oper
        ndim = len(oper.axes)
        if ndim == 1:
            phase = 0.5 * oper.deltax * oper.kx
        elif ndim == 2:
            phase = 0.5 * (oper.deltax * oper.KX + oper.deltay * oper.KY)
        elif ndim == 3:
            phase = 0.5 * (
                oper.deltax * oper.Kx
                + oper.deltay * oper.Ky
                + oper.deltaz * oper.Kz
            )
        else:
            raise NotImplementedError
        self._phaseshift = np.exp(1j * phase)
        return self._phaseshift

    def _init_phaseshift_random(self):

        params_phaseshift = self.params.time_stepping.phaseshift_random
        if params_phaseshift.nb_steps_compute_new_pair is None:
            if params_phaseshift.nb_pairs == 1:
                params_phaseshift.nb_steps_compute_new_pair = 2
            else:
                params_phaseshift.nb_steps_compute_new_pair = (
                    4 * params_phaseshift.nb_pairs
                )

        self._index_phaseshift = 1
        self._previous_index_pair = 0
        self._previous_index_flip = 0
        self._pairs_phaseshift = []
        for _ in range(params_phaseshift.nb_pairs):
            phaseshift_alpha = np.empty(
                self.sim.oper.shapeK_loc, dtype=np.complex128
            )
            phaseshift_beta = np.empty_like(phaseshift_alpha)
            phase_alpha, phase_beta = self.sim.oper.get_phases_random()

            self._pairs_phaseshift.append(
                compute_phaseshift_terms(
                    phase_alpha, phase_beta, phaseshift_alpha, phaseshift_beta
                )
            )

    def _get_phaseshift_random(self):
        """Compute two random phase shift terms."""
        params_phaseshift = self.params.time_stepping.phaseshift_random

        nb_pairs = params_phaseshift.nb_pairs
        nb_steps_compute_new_pair = params_phaseshift.nb_steps_compute_new_pair

        if nb_pairs == 1 and nb_steps_compute_new_pair == 1:
            phaseshift_alpha, phaseshift_beta = self._pairs_phaseshift[0]
        elif nb_pairs == 1 and nb_steps_compute_new_pair == 2:
            pair = self._pairs_phaseshift[0]
            if self._index_phaseshift == 1:
                phaseshift_alpha, phaseshift_beta = pair
            else:
                phaseshift_beta, phaseshift_alpha = pair
        else:
            index_pair = randint(0, nb_pairs - 1)
            pair = self._pairs_phaseshift[index_pair]
            index_flip = randint(0, 1)
            if (
                index_pair == self._previous_index_pair
                and index_flip == self._previous_index_flip
            ):
                index_flip = 0 if index_flip else 1
            self._previous_index_pair = index_pair
            self._previous_index_flip = index_flip
            if index_flip:
                phaseshift_alpha, phaseshift_beta = pair
            else:
                phaseshift_beta, phaseshift_alpha = pair

        if self._index_phaseshift == nb_steps_compute_new_pair:
            self._index_phaseshift = 1
            phase_alpha, phase_beta = self.sim.oper.get_phases_random()
            phaseshift_alpha, phaseshift_beta = self._pairs_phaseshift.pop(0)
            self._pairs_phaseshift.append(
                compute_phaseshift_terms(
                    phase_alpha, phase_beta, phaseshift_alpha, phaseshift_beta
                )
            )
        else:
            self._index_phaseshift += 1

        return phaseshift_alpha, phaseshift_beta

    def _time_step_Euler_phaseshift(self):
        r"""Forward Euler method, dealiasing with phase-shifting.

        Notes
        -----

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
        phaseshift = self._get_phaseshift()
        tendencies_shifted = (
            compute_tendencies(phaseshift * state_spect) / phaseshift
        )
        # dealiased tendencies
        tendencies_dealiased = 0.5 * (tendencies_0 + tendencies_shifted)
        step_Euler_inplace(state_spect, dt, tendencies_dealiased, diss)

    def _time_step_Euler_phaseshift_random(self):
        r"""Forward Euler method, dealiasing with phase-shifting.

        Notes
        -----

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
        phaseshift_alpha, phaseshift_beta = self._get_phaseshift_random()
        tendencies_alpha = (
            compute_tendencies(phaseshift_alpha * state_spect) / phaseshift_alpha
        )
        tendencies_beta = (
            compute_tendencies(phaseshift_beta * state_spect) / phaseshift_beta
        )
        # dealiased tendencies
        tendencies_dealiased = 0.5 * (tendencies_alpha + tendencies_beta)
        step_Euler_inplace(state_spect, dt, tendencies_dealiased, diss)

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
        step_Euler(
            state_spect, dt / 2, tendencies_0, diss2, output=state_spect_12
        )

        tendencies_12 = compute_tendencies(state_spect_12, old=tendencies_0)
        step_like_RK2(state_spect, dt, tendencies_12, diss, diss2)

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
        step_Euler(state_spect, dt, tendencies_0, diss, output=state_spect_1)

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
        step_Euler(state_spect, dt, tendencies_0, diss, output=state_spect_1)

        phaseshift = self._get_phaseshift()
        tendencies_1_shift = compute_tendencies(phaseshift * state_spect_1)

        tendencies_d = self._state_spect_tmp
        if ts.is_transpiled:
            ts.use_block("rk2_tendencies_d")
        else:
            # transonic block (
            #     A tendencies_d, tendencies_0, tendencies_1_shift;
            #     Am1 phaseshift;
            # )
            tendencies_d[:] = 0.5 * (
                tendencies_0 + tendencies_1_shift / phaseshift
            )

        step_like_RK2(state_spect, dt, tendencies_d, diss, diss2)

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

          .. math:: \p_t \log S = \sigma + \frac{\tilde N_{0\alpha}}{S_0},

          Integrating from :math:`t` to :math:`t+\dt`, it gives:

          .. math:: S_1 = (S_0 + \tilde N_{0\alpha} \dt) e^{\sigma \dt}.

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

        phaseshift_alpha, phaseshift_beta = self._get_phaseshift_random()

        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        tmp0 = np.empty_like(state_spect)
        state_spect_shift = mul(phaseshift_alpha, state_spect, output=tmp0)
        tendencies_0_shift = compute_tendencies(
            state_spect_shift, old=state_spect_shift
        )

        tendencies_0 = div_inplace(tendencies_0_shift, phaseshift_alpha)

        state_spect_1 = step_Euler(
            state_spect, dt, tendencies_0, diss, output=self._state_spect_tmp
        )

        tmp1 = np.empty_like(state_spect)
        state_spect_1_shift = mul(phaseshift_beta, state_spect_1, output=tmp1)
        tendencies_1_shift = compute_tendencies(
            state_spect_1_shift, old=state_spect_1_shift
        )

        tendencies_d = mean_with_phaseshift(
            tendencies_0,
            tendencies_1_shift,
            phaseshift_beta,
            output=self._state_spect_tmp,
        )
        step_like_RK2(state_spect, dt, tendencies_d, diss, diss2)

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
        phaseshift = self._get_phaseshift()
        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        tmp0 = np.empty_like(state_spect)
        tmp1 = np.empty_like(state_spect)
        tmp2 = np.empty_like(state_spect)
        tmp3 = np.empty_like(state_spect)

        tendencies_0 = compute_tendencies(state_spect, old=tmp0)
        state_spect_shift = mul(phaseshift, state_spect, output=tmp1)
        tendencies_0_shift = compute_tendencies(
            state_spect_shift, old=state_spect_shift
        )
        tendencies_d0 = mean_with_phaseshift(
            tendencies_0, tendencies_0_shift, phaseshift, output=tmp2
        )

        state_spect_1 = step_Euler(
            state_spect, dt, tendencies_d0, diss, output=self._state_spect_tmp
        )

        tendencies_1 = compute_tendencies(state_spect_1, old=tmp0)
        state_spect_shift = mul(phaseshift, state_spect_1, output=tmp1)
        tendencies_1_shift = compute_tendencies(
            state_spect_shift, old=state_spect_shift
        )

        tendencies_d = tmp3
        if ts.is_transpiled:
            ts.use_block("rk2_exact")
        else:
            # based on approximation 1
            # transonic block (
            #     A tendencies_d, tendencies_d0, tendencies_1, tendencies_1_shift;
            #     Am1 phaseshift
            # )

            tendencies_d[:] = 0.5 * (
                tendencies_d0
                + 0.5 * (tendencies_1 + tendencies_1_shift / phaseshift)
            )

        step_like_RK2(state_spect, dt, tendencies_d, diss, diss2)

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

          .. math:: \p_t \log S = \sigma + \frac{N_0}{S_0},

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
             \frac{1}{2} N_0 e^{\sigma \dt}
             + N(\SA1dt2) e^{\sigma \frac{\dt}{2}}
             + N(\SA2dt2) e^{\sigma \frac{\dt}{2}}
             + \frac{1}{2} N(S_{A3\dt})\right].

        """
        dt = self.deltat
        diss, diss2 = self.exact_linear_coefs.get_updated_coefs()

        compute_tendencies = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        tendencies_0 = compute_tendencies()
        state_spect_tmp1 = self._state_spect_tmp1

        # rk4_step0
        state_spect_tmp = step_Euler(
            state_spect, dt / 6, tendencies_0, diss, output=self._state_spect_tmp
        )
        state_spect_12_approx1 = step_Euler(
            state_spect,
            dt / 2,
            tendencies_0,
            diss2,
            output=state_spect_tmp1,
        )

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
