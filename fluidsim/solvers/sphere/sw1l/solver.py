"""SW1L equations over a sphere (:mod:`fluidsim.solvers.sphere.sw1l`)
=====================================================================

.. autoclass:: InfoSolverSphereSW1L
   :members:
   :private-members:

.. autoclass:: SimulSphereSW1L
   :members:
   :private-members:

"""

from fluidsim.base.sphericalharmo.solver import (
    InfoSolverSphericalHarmo,
    SimulSphericalHarmo,
)

from fluidsim.solvers.sw1l.solver import (
    compute_Frot,
    SetOfVariables,
    compute_tendencies_nonlin_sw1l,
)


class InfoSolverSphereSW1L(InfoSolverSphericalHarmo):
    """Contain the information on a base pseudo-spectral solver."""

    def _init_root(self):
        """Init. `self` by writting the information on the solver.
        """

        super(InfoSolverSphereSW1L, self)._init_root()

        here = "fluidsim.solvers.sphere.sw1l"

        self.module_name = here + ".solver"
        self.class_name = "SimulSphereSW1L"
        self.short_name = "sphere.sw1l"


# self.classes.State.module_name = here + '.state'
# self.classes.State.class_name = 'StateSphericalHarmo'

# self.classes.Output.module_name = here + '.output'
# self.classes.Output.class_name = 'Output'


class SimulSphereSW1L(SimulSphericalHarmo):
    """Pseudo-spectral base solver."""

    InfoSolver = InfoSolverSphereSW1L

    # @staticmethod
    # def _complete_params_with_default(params):
    #     """Complete the `params` container (static method)."""
    #     SimulSphericalHarmo._complete_params_with_default(params)
    #     # Coriolis parameter (not implemented)
    #     attribs = {'f': 0.}
    #     params._set_attribs(attribs)

    def tendencies_nonlin(self, state_spect=None, old=None):
        r"""Compute the nonlinear tendencies.

        Parameters
        ----------

        state_spect : :class:`fluidsim.base.setofvariables.SetOfVariables`
            optional

            Array containing the state, i.e. the vorticity, in Fourier
            space.  If `state_spect`, the variables vorticity and the
            velocity are computed from it, otherwise, they are taken
            from the global state of the simulation, `self.state`.

            These two possibilities are used during the Runge-Kutta
            time-stepping.

        Returns
        -------

        tendencies_sh : :class:`fluidsim.base.setofvariables.SetOfVariables`
            An array containing the tendencies for the vorticity.

        Notes
        -----

        .. |p| mathmacro:: \partial

        The 2D Navier-Stokes equation can be written

        .. math:: \p_t \hat\zeta = \hat N(\zeta) - \sigma(k) \hat \zeta,

        This function compute the nonlinear term ("tendencies")
        :math:`N(\zeta) = - \mathbf{u}\cdot \mathbf{\nabla} \zeta`.

        """
        # the operator and the fast Fourier transform
        oper = self.oper

        # get or compute rot_sh, ux and uy
        if state_spect is None:
            rot_sh = self.state.state_spect.get_var("rot_sh")
            ux = self.state.state_phys.get_var("ux")
            uy = self.state.state_phys.get_var("uy")
        else:
            rot_sh = state_spect.get_var("rot_sh")
            ux, uy = oper.vec_from_rotsh(rot_sh)

        # "px" like $\partial_x$
        px_rot, py_rot = oper.gradf_from_fsh(rot_sh)

        Frot = compute_Frot(ux, uy, px_rot, py_rot, beta=0.0)

        if old is None:
            tendencies_sh = SetOfVariables(like=self.state.state_spect)
        else:
            tendencies_sh = old
        Frot_sh = tendencies_sh.get_var("rot_sh")
        oper.sht_as_arg(Frot, Frot_sh)

        # oper.dealiasing(Frot_sh)

        import numpy as np

        T_rot = np.real(Frot_sh.conj() * rot_sh + Frot_sh * rot_sh.conj()) / 2.0
        print(
            ("sum(T_rot) = {0:9.4e} ; sum(abs(T_rot)) = {1:9.4e}").format(
                self.oper.sum_wavenumbers(T_rot),
                self.oper.sum_wavenumbers(abs(T_rot)),
            )
        )

        if self.params.forcing.enable:
            tendencies_sh += self.forcing.get_forcing()

        return tendencies_sh

        ## The NS2D implementation ends here

        oper = self.oper

        if state_spect is None:
            state_phys = self.state.state_phys
        else:
            state_phys = self.state.return_statephys_from_statespect(state_spect)

        ux = state_phys.get_var("ux")
        uy = state_phys.get_var("uy")
        eta = state_phys.get_var("eta")
        rot = state_phys.get_var("rot")

        if old is None:
            tendencies_sh = SetOfVariables(
                like=self.state.state_spect, info="tendencies_nonlin"
            )
        else:
            tendencies_sh = old

        Fx_sh = tendencies_sh.get_var("ux_sh")
        Fy_sh = tendencies_sh.get_var("uy_sh")
        Feta_sh = tendencies_sh.get_var("eta_sh")

        compute_tendencies_nonlin_sw1l(
            rot,
            ux,
            uy,
            eta,
            Fx_sh,
            Fy_sh,
            Feta_sh,
            self.params.f,
            self.params.c2,
            oper.sht,
            oper.gradf_from_fsh,  # Warning comments in implemenetation
            oper.dealiasing,  # Barely Implemented, simply returns as is
            oper.divfft_from_vecfft,  # NotImplemted
        )

        oper.dealiasing(tendencies_sh)

        if self.params.forcing.enable:
            tendencies_sh += self.forcing.get_forcing()

        return tendencies_sh


Simul = SimulSphereSW1L
