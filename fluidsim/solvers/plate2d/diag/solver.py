# -*- coding: utf-8 -*-

"""Plate2d solver diag. (:mod:`fluidsim.solvers.plate2d.diag.solver`)
===========================================================================

.. currentmodule:: fluidsim.solvers.plate2d.diag.solver

Provides:

.. autoclass:: Simul
   :members:
   :private-members:

"""

from __future__ import print_function

import numpy as np

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.base.solvers.pseudo_spect import SimulBasePseudoSpectral
from fluidsim.solvers.plate2d.solver import InfoSolverPlate2D


class InfoSolverPlate2DDiag(InfoSolverPlate2D):
    def _init_root(self):

        super(InfoSolverPlate2DDiag, self)._init_root()

        self.short_name = 'plate2d.diag'

        package = 'fluidsim.solvers.' + self.short_name

        classes = self.classes

        classes.State.module_name = package + '.state'
        classes.State.class_name = 'StatePlate2DDiag'

        classes.Forcing.module_name = package + '.forcing'
        classes.Forcing.class_name = 'ForcingPlate2DDiag'


class Simul(SimulBasePseudoSpectral):
    r"""Pseudo-spectral solver solving the Föppl-von Kármán equations.

    Notes
    -----

    .. |p| mathmacro:: \partial

    This class is dedicated to solve with a pseudo-spectral method the
    Föppl-von Kármán equations which describe the dynamics of a rigid
    plate (see :class:`fluidsim.solvers.plate2d.solver.Simul`).

    In the Fourier space, the governing equations write:

    .. math::
       \p_t \hat z = \hat w,

       \p_t \hat w = - \Omega(k)^2 \hat z + \widehat{N_w(z)} + \hat F_w
       - \gamma_w \hat w,

    where :math:`\Omega(k) = k^4`, :math:`k^2 = |\mathbf{k}|^2` and
    :math:`\gamma_w = \nu_\alpha k^{2\alpha}`. For this solver, we
    will use variables that diagonalized the linear terms, i.e. that
    represent propagative waves (see
    :class:`fluidsim.solvers.waves2d.solver.Simul`). Therefore,
    all the linear terms can be solved exactly. Applying the equations
    of :class:`fluidsim.solvers.waves2d.solver.Simul` with
    :math:`\gamma_f = \gamma_z = 0`, we find that the eigenvalues are
    :math:`\sigma_\pm = - \gamma_w/2 \pm i \tilde \Omega` with
    :math:`\tilde \Omega = \Omega \sqrt{1 - (\gamma_w/(2 \Omega))^2}`,
    and the (not normalized) eigenvectors are

    .. math::

       V_\pm = \begin{pmatrix} 1 \\ \sigma_\pm \end{pmatrix}.

    The state can be represented by a vector :math:`A` that verifies
    :math:`X = V A`, where :math:`V` is the base matrix

    .. math::

       V = \begin{pmatrix} 1 & 1 \\
       \sigma_+ & \sigma_- \end{pmatrix}.

    The inverse base matrix is given by

    .. math::

       V^{-1} = \frac{i}{2\tilde \Omega}
       \begin{pmatrix}
       \sigma_- & -1 \\
       -\sigma_+ &  1 \end{pmatrix},

    which gives more explicitly that

    .. math::

       A = \begin{pmatrix} \hat a_+ \\ \hat a_- \end{pmatrix} =
       \frac{i}{2\tilde \Omega} \begin{pmatrix} \sigma_- \hat z - \hat
       w \\ -\sigma_+ \hat z + \hat w \end{pmatrix}.

    The governing equations can then be expressed as

    .. math::

       \p_t A = L A + N(A),

    with


    .. math::

       L = \begin{pmatrix} \sigma_+ & 0 \\ 0 & \sigma_- \end{pmatrix},

       N = \frac{i}{2\tilde \Omega} \begin{pmatrix} -(\widehat{N_w(z)}
       + \hat F_w) \\ \widehat{N_w(z)} + \hat F_w \end{pmatrix}.

    """
    InfoSolver = InfoSolverPlate2DDiag

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        SimulBasePseudoSpectral._complete_params_with_default(params)
        attribs = {'beta': 0.}
        params._set_attribs(attribs)

    def tendencies_nonlin(self, state_fft=None):
        """Compute the "nonlinear" tendencies."""
        oper = self.oper

        if state_fft is None:
            state_fft = self.state.state_fft

        # w_fft = state_fft.get_var('w_fft')
        z_fft = state_fft.get_var('z_fft')

        mamp_zz = oper.monge_ampere_from_fft(z_fft, z_fft)
        chi_fft = - oper.invlaplacian2_fft(oper.fft2(mamp_zz))
        mamp_zchi = oper.monge_ampere_from_fft(z_fft, chi_fft)
        Nw_fft = oper.fft2(mamp_zchi)

        if self.params.FORCING:
            # this is WRONG
            forcing_fft = self.forcing.get_forcing()
            forcing_w_fft = forcing_fft
            Nw_fft += forcing_w_fft

        oper.dealiasing(Nw_fft)

        tendencies_fft = SetOfVariables(
            like=self.state.state_fft,
            info='tendencies_nonlin')

        tendencies_fft.set_var('ap_fft', -Nw_fft)
        tendencies_fft.set_var('am_fft', Nw_fft)

        tendencies_fft /= -2j*self._tilde_Omega

        # ratio = self.test_tendencies_nonlin(
        #     tendencies_fft, w_fft, z_fft, chi_fft)
        # print('ratio:', ratio)

        return tendencies_fft

    def compute_freq_diss(self):
        """Compute the dissipation frequencies with dissipation only for w."""
        f_d_w, f_d_hypo_w = super(Simul, self).compute_freq_diss()
        f_d = np.zeros_like(self.state.state_fft, dtype=np.float64)
        f_d_hypo = np.zeros_like(self.state.state_fft,
                                 dtype=np.float64)
        f_d[0] = f_d_w
        f_d_hypo[0] = f_d_hypo_w
        return f_d, f_d_hypo

    # def test_tendencies_nonlin(
    #         self, tendencies_fft=None,
    #         w_fft=None, z_fft=None, chi_fft=None):
    #     r"""Test if the tendencies conserves the total energy.

    #     We consider the conservative Föppl-von Kármán equations
    #     (without dissipation and forcing) written as

    #     .. math::

    #        \p_t z = F_z

    #        \p_t w = F_w

    #     We have:

    #     .. math::

    #        \p_t E_K(\mathbf{k}) = \mathcal{R} ( \hat F_w \hat w ^* )

    #        \p_t E_L(\mathbf{k}) = k^4 \mathcal{R} ( \hat F_z \hat z ^* )

    #        \p_t E_{NQ}(\mathbf{k}) =
    #        - \mathcal{R} ( \widehat{\{ F_z, z\}} \hat \chi ^* )

    #     Since the total energy is conserved, we should have

    #     .. math::

    #        \sum_{\mathbf{k}} \p_t E_K(\mathbf{k}) + \p_t E_L(\mathbf{k})
    #        + \p_t E_{NQ}(\mathbf{k}) = 0

    #     This function computes this quantities.

    #     """

    #     if tendencies_fft is None:
    #         tendencies_fft = self.tendencies_nonlin()
    #         w_fft = self.state.state_fft['w_fft']
    #         z_fft = self.state.state_fft['z_fft']
    #         chi_fft = self.state.compute('chi_fft')

    #     F_w_fft = tendencies_fft['w_fft']
    #     F_z_fft = tendencies_fft['z_fft']

    #     K4 = self.oper.K4

    #     dt_E_K = np.real(F_w_fft * w_fft.conj())
    #     dt_E_L = K4 * np.real(F_z_fft * z_fft.conj())

    #     tmp = self.oper.monge_ampere_from_fft(F_z_fft, z_fft)
    #     tmp_fft = self.oper.fft2(tmp)

    #     dt_E_NQ = - np.real(tmp_fft * chi_fft.conj())

    #     T = dt_E_K + dt_E_L + dt_E_NQ

    #     norm = self.oper.sum_wavenumbers(abs(T))

    #     if norm < 1e-15:
    #         print('Only zeros in total energy tendency.')
    #         # print('(K+L)\n', dt_E_K+dt_E_L)
    #         # print('NQ\n', dt_E_NQ)
    #         return 0
    #     else:
    #         T = T/norm
    #         # print('ratio array\n', T)
    #         # print('(K+L)\n', (dt_E_K+dt_E_L)/norm)
    #         # print('NQ\n', dt_E_NQ/norm)
    #         return self.oper.sum_wavenumbers(T)


if __name__ == "__main__":

    np.set_printoptions(precision=2)

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    nh = 192/2
    Lh = 2*np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh
    # params.oper.type_fft = 'FFTWPY'
    params.oper.coef_dealiasing = 2./3

    delta_x = params.oper.Lx/params.oper.nx
    params.nu_8 = 2.*10e-4*params.forcing.forcing_rate**(1./3)*delta_x**8

    kmax = np.sqrt(2)*np.pi/delta_x

    params.time_stepping.USE_CFL = False
    params.time_stepping.deltat0 = 2*np.pi/kmax**2
    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = 50.0
    params.time_stepping.it_end = 1

    # params.init_fields.type = 'HARMONIC'
    params.init_fields.type = 'noise'
    params.init_fields.max_velo_noise = 0.001
    # params.init_fields.path_file = (
    #     '/home/users/bonamy2c/Sim_data/PLATE2D_test_L='
    #     '2pix2pi_256x256_2015-03-04_22-36-37/state_phys_t=000.100.hd5')

    params.FORCING = True
    params.forcing.forcing_rate = 100.
    # params.forcing.nkmax_forcing = 5
    # params.forcing.nkmin_forcing = 4

    params.output.periods_print.print_stdout = 0.5

    params.output.periods_save.phys_fields = 0.0
    params.output.periods_save.spectra = 0.5
    # params.output.periods_save.spect_energy_budg = 0.5
    # params.output.periods_save.increments = 0.5

    params.output.ONLINE_PLOT_OK = False
    params.output.period_refresh_plots = 0.5
    params.output.periods_plot.phys_fields = 0.0

    params.output.phys_fields.field_to_plot = 'z'

    params.output.spectra.HAS_TO_PLOT_SAVED = True

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    sim.output.phys_fields.plot()

    fld.show()
