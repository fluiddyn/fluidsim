"""NS2D equations over a sphere (:mod:`fluidsim.solvers.sphere.ns2d`)
=====================================================================

.. autoclass:: InfoSolverSphereNS2D
   :members:
   :private-members:

.. autoclass:: SimulSphereNS2D
   :members:
   :private-members:

"""

from fluidsim.base.sphericalharmo.solver import (
    InfoSolverSphericalHarmo,
    SimulSphericalHarmo,
)

from fluidsim.solvers.ns2d.solver import compute_Frot, SetOfVariables


class InfoSolverSphereNS2D(InfoSolverSphericalHarmo):
    """Contain the information on a base pseudo-spectral solver."""

    def _init_root(self):
        """Init. `self` by writting the information on the solver."""

        super()._init_root()

        here = "fluidsim.solvers.sphere.ns2d"

        self.module_name = here + ".solver"
        self.class_name = "SimulSphereNS2D"
        self.short_name = "sphere.ns2d"


# self.classes.State.module_name = here + '.state'
# self.classes.State.class_name = 'StateSphericalHarmo'

# self.classes.Output.module_name = here + '.output'
# self.classes.Output.class_name = 'Output'


class SimulSphereNS2D(SimulSphericalHarmo):
    """Pseudo-spectral base solver."""

    InfoSolver = InfoSolverSphereNS2D

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

        T_rot = (Frot_sh.conj() * rot_sh).real
        print(
            ("sum(T_rot) = {:9.4e} ; sum(abs(T_rot)) = {:9.4e}").format(
                self.oper.sum_wavenumbers(T_rot),
                self.oper.sum_wavenumbers(abs(T_rot)),
            )
        )

        if self.params.forcing.enable:
            tendencies_sh += self.forcing.get_forcing()

        return tendencies_sh


Simul = SimulSphereNS2D

if __name__ == "__main__":

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    params.init_fields.type = "noise"

    params.time_stepping.USE_CFL = True
    params.time_stepping.t_end = 10.0
    # params.time_stepping.deltat0 = 0.1

    params.output.periods_save.phys_fields = 0.25

    sim = Simul(params)
    sim.time_stepping.start()
