"""SW1L equations over a sphere (:mod:`fluidsim.solvers.sphere.sw1l`)
=====================================================================

.. autoclass:: InfoSolverSphereSW1L
   :members:
   :private-members:

.. autoclass:: SimulSphereSW1L
   :members:
   :private-members:

"""
from fluidpythran import cachedjit
from fluidsim.base.sphericalharmo.solver import (
    InfoSolverSphericalHarmo,
    SimulSphericalHarmo,
)

from fluidsim.solvers.sw1l.solver import (
    compute_Frot,
    SetOfVariables,
    compute_tendencies_nonlin_sw1l,
)


Af = "float64[:, :]"


@cachedjit
def compute_Frot(rot: Af, ux: Af, uy: Af, f_radial: Af):
    """Compute cross-product of absolute potential vorticity with velocity."""
    rot_abs = rot + f_radial
    F1x = rot_abs * uy
    F1y = -rot_abs * ux

    return F1x, F1y


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

        self.classes.State.module_name = here + ".state"
        self.classes.State.class_name = "StateSphericalHarmoSW1L"


# self.classes.Output.module_name = here + '.output'
# self.classes.Output.class_name = 'Output'


class SimulSphereSW1L(SimulSphericalHarmo):
    """Pseudo-spectral base solver."""

    InfoSolver = InfoSolverSphereSW1L

    @staticmethod
    def _complete_params_with_default(params):
        """Complete the `params` container (static method). 

        """
        SimulSphericalHarmo._complete_params_with_default(params)

        attribs = {"c2": 20}
        params._set_attribs(attribs)

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

        if state_spect is None:
            state_spect = self.state.state_spect
            state_phys = self.state.state_phys
        else:
            state_phys = self.state.return_statephys_from_statespect(state_spect)

        # get or compute rot_sh, ux and uy
        ux = state_phys.get_var("ux")
        uy = state_phys.get_var("uy")
        eta = state_phys.get_var("eta")
        rot = state_phys.get_var("rot")

        if old is None:
            tendencies_sh = SetOfVariables(like=self.state.state_spect)
        else:
            tendencies_sh = old

        Frot_sh = tendencies_sh.get_var("rot_sh")
        Fdiv_sh = tendencies_sh.get_var("div_sh")
        Feta_sh = tendencies_sh.get_var("eta_sh")
        c2 = self.params.c2

        # Absolute vorticity
        Fux, Fuy = compute_Frot(rot, ux, uy, oper.f_radial)
        oper.divrotsh_from_vec(Fux, Fuy, Fdiv_sh, Frot_sh)

        # TODO: Pythranize: laplacian
        # Subtract laplacian of total energy K.E. + A.P.E from divergence tendency
        Fdiv_sh += oper.K2 * oper.sht(0.5 * (ux ** 2 + uy ** 2) + c2 * eta)

        # Calculate Feta_sh = \nabla.(hu) = \nabla.((1 + \eta)u)
        oper.divrotsh_from_vec(-(eta + 1) * ux, -(eta + 1) * uy, Feta_sh)

        # oper.dealiasing(tendencies_sh)

        # def check_conservation(Fvar_sh, var_sh, var_str):
        #     import numpy as np
        #     T = np.real(Fvar_sh.conj() * var_sh + Fvar_sh * var_sh.conj()) / 2.0
        #     print(
        #         ("sum(T_{2}) = {0:9.4e} ; sum(abs(T_{2})) = {1:9.4e}").format(
        #             self.oper.sum_wavenumbers(T),
        #             self.oper.sum_wavenumbers(abs(T)),
        #             var_str
        #         ),
        #         end =" | "
        #     )
        # rot_sh = state_spect.get_var("rot_sh")
        # div_sh = state_spect.get_var("div_sh")
        # check_conservation(Frot_sh, rot_sh, "rot")
        # check_conservation(Fdiv_sh, div_sh, "div")
        # print()

        if self.params.forcing.enable:
            tendencies_sh += self.forcing.get_forcing()

        return tendencies_sh


Simul = SimulSphereSW1L


if __name__ == "__main__":

    params = Simul.create_default_params()

    params.short_name_type_run = "test"
    params.output.sub_directory = "test"

    params.nu_2 = 1e-2
    params.oper.radius = 10.0
    params.oper.omega = 0.0
    params.forcing.enable = False

    params.init_fields.type = "noise"
    params.time_stepping.deltat0 = 1e-3
    params.time_stepping.USE_CFL = False
    params.time_stepping.t_end = 10.0
    # params.time_stepping.deltat0 = 0.1

    params.output.periods_save.phys_fields = 0.25

    sim = Simul(params)
    sim.time_stepping.start()
