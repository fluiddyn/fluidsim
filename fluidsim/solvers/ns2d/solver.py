"""NS2D solver (:mod:`fluidsim.solvers.ns2d.solver`)
====================================================

This module provides two classes defining the pseudo-spectral solver
2D incompressible Navier-Stokes equations (ns2d).

.. autoclass:: InfoSolverNS2D
   :members:
   :private-members:

.. autoclass:: Simul
   :members:
   :private-members:

"""

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral, InfoSolverPseudoSpectral)


class InfoSolverNS2D(InfoSolverPseudoSpectral):
    """Contain the information on the solver ns2d."""
    def _init_root(self):
        """Init. `self` by writting the information on the solver.

        The function `InfoSolverPseudoSpectral._init_root` is
        called. We keep two classes listed by this function:

        - :class:`fluidsim.base.time_stepping.pseudo_spect_cy.TimeSteppingPseudoSpectral`

        - :class:`fluidsim.operators.operators.OperatorsPseudoSpectral2D`

        The other first-level classes for this solver are:

        - :class:`fluidsim.solvers.ns2d.solver.Simul`

        - :class:`fluidsim.solvers.ns2d.state.StateNS2D`

        - :class:`fluidsim.solvers.ns2d.init_fields.InitFieldsNS2D`

        - :class:`fluidsim.solvers.ns2d.output.Output`

        - :class:`fluidsim.solvers.ns2d.forcing.ForcingNS2D`

        """
        super(InfoSolverNS2D, self)._init_root()

        package = 'fluidsim.solvers.ns2d'
        self.module_name = package + '.solver'
        self.class_name = 'Simul'
        self.short_name = 'NS2D'

        classes = self.classes

        classes.State.module_name = package + '.state'
        classes.State.class_name = 'StateNS2D'

        classes.InitFields.module_name = package + '.init_fields'
        classes.InitFields.class_name = 'InitFieldsNS2D'

        classes.Output.module_name = package + '.output'
        classes.Output.class_name = 'Output'

        classes.Forcing.module_name = package + '.forcing'
        classes.Forcing.class_name = 'ForcingNS2D'


class Simul(SimulBasePseudoSpectral):
    """Pseudo-spectral solver 2D incompressible Navier-Stokes equations.

    """
    InfoSolver = InfoSolverNS2D

    @staticmethod
    def _complete_params_with_default(params):
        """Complete the `params` container (static method)."""
        SimulBasePseudoSpectral._complete_params_with_default(params)
        attribs = {'beta': 0.}
        params._set_attribs(attribs)

    def tendencies_nonlin(self, state_fft=None):
        r"""Compute the nonlinear tendencies.

        Parameters
        ----------

        state_fft : :class:`fluidsim.base.setofvariables.SetOfVariables`
            optional

            Array containing the state, i.e. the vorticity, in Fourier
            space.  If `state_fft`, the variables vorticity and the
            velocity are computed from it, otherwise, they are taken
            from the global state of the simulation, `self.state`.

            These two possibilities are used during the Runge-Kutta
            time-stepping.

        Returns
        -------

        tendencies_fft : :class:`fluidsim.base.setofvariables.SetOfVariables`
            An array containing the tendencies for the vorticity.

        Notes
        -----

        .. |p| mathmacro:: \partial

        The 2D Navier-Stockes equation can be written

        .. math:: \p_t \hat\zeta = \hat N(\zeta) - \sigma(k) \zeta,

        This function compute the nonlinear term ("tendencies")
        :math:`\hat N(\zeta) = - \mathbf{u}\cdot \mathbf{\nabla}
        \zeta`.

        """
        # the operator and the fast Fourier transform
        oper = self.oper
        fft2 = oper.fft2
        ifft2 = oper.ifft2

        # get or compute rot_fft, ux and uy
        if state_fft is None:
            rot_fft = self.state.state_fft.get_var('rot_fft')
            ux = self.state.state_phys.get_var('ux')
            uy = self.state.state_phys.get_var('uy')
        else:
            rot_fft = state_fft.get_var('rot_fft')
            ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
            ux = ifft2(ux_fft)
            uy = ifft2(uy_fft)

        # "px" like $\partial_x$
        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_rot = ifft2(px_rot_fft)
        py_rot = ifft2(py_rot_fft)

        if self.params.beta == 0:
            Frot = -ux*px_rot - uy*py_rot
        else:
            Frot = -ux*px_rot - uy*(py_rot + self.params.beta)

        Frot_fft = fft2(Frot)
        oper.dealiasing(Frot_fft)

        # T_rot = np.real(Frot_fft.conj()*rot_fft
        #                + Frot_fft*rot_fft.conj())/2.
        # print ('sum(T_rot) = {0:9.4e} ; sum(abs(T_rot)) = {1:9.4e}'
        #       ).format(self.oper.sum_wavenumbers(T_rot),
        #                self.oper.sum_wavenumbers(abs(T_rot)))

        tendencies_fft = SetOfVariables(like=self.state.state_fft)
        tendencies_fft.set_var('rot_fft', Frot_fft)

        if self.params.FORCING:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


if __name__ == "__main__":

    from math import pi

    # import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    params.oper.nx = params.oper.ny = nh = 32
    params.oper.Lx = params.oper.Ly = Lh = 2 * pi

    delta_x = Lh / nh

    params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8

    params.time_stepping.t_end = 1.

    params.init_fields.type = 'dipole'

    params.FORCING = True
    params.forcing.type = 'random'
    # 'Proportional'
    # params.forcing.type_normalize

    params.output.sub_directory = 'tests'

    # params.output.periods_print.print_stdout = 0.25

    params.output.periods_save.phys_fields = 1.
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spatial_means = 0.05
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.periods_save.increments = 0.5

    params.output.periods_plot.phys_fields = 0.0

    params.output.ONLINE_PLOT_OK = True

    # params.output.spectra.HAS_TO_PLOT_SAVED = True
    # params.output.spatial_means.HAS_TO_PLOT_SAVED = True
    # params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True
    # params.output.increments.HAS_TO_PLOT_SAVED = True

    params.output.phys_fields.field_to_plot = 'rot'

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    # fld.show()
