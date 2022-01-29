"""One-layer shallow-water (Saint Venant) equations solver (:mod:`fluidsim.solvers.sw1l.solver`)
================================================================================================

This module provides two classes defining the pseudo-spectral solver for
biperiodic one-layer shallow water equations, expressed using primitive
variables, i.e. the horizontal velocities and surface dispacement.

.. autoclass:: InfoSolverSW1L
   :members:
   :private-members:

.. autoclass:: Simul
   :members:
   :private-members:



"""

from transonic import jit, Array
from fluiddyn.util import mpi

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral,
    InfoSolverPseudoSpectral,
)

A = Array[float, "2d"]


@jit
def compute_Frot(rot: A, ux: A, uy: A, f: float):
    """Compute cross-product of absolute potential vorticity with velocity."""
    if f != 0:
        rot_abs = rot + f
    else:
        rot_abs = rot

    F1x = rot_abs * uy
    F1y = -rot_abs * ux

    return F1x, F1y


@jit
def compute_pressure(c2: float, eta: A, ux: A, uy: A):
    return c2 * eta + 0.5 * (ux**2 + uy**2)


class InfoSolverSW1L(InfoSolverPseudoSpectral):
    """Information about the solver SW1L."""

    def _init_root(self):
        """The simulation object is instantiated with classes defined in this
        function.

        The super function `InfoSolverPseudoSpectral._init_root` is
        called first. A few classes defined by this function are retained as it
        is:

        - :class:`fluidsim.base.time_stepping.pseudo_spect.TimeSteppingPseudoSpectral`

        - :class:`fluidsim.solvers.sw1l.OperatorsPseudoSpectralSW1L`

        - :class:`fluidsim.base.preprocess.pseudo_spect.PreprocessPseudoSpectral`

        Solver-specific first-level classes are added:

        - :class:`fluidsim.solvers.sw1l.solver.Simul`

        - :class:`fluidsim.solvers.sw1l.state.StateSW1L`

        - :class:`fluidsim.solvers.sw1l.init_fields.InitFieldsSW1L`

        - :class:`fluidsim.solvers.sw1l.output.OutputSW1L`

        - :class:`fluidsim.solvers.sw1l.forcing.ForcingSW1L`

        """
        super()._init_root()

        package = "fluidsim.solvers.sw1l"

        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "SW1L"

        classes = self.classes

        classes.Operators.module_name = package + ".operators"
        classes.Operators.class_name = "OperatorsPseudoSpectralSW1L"

        classes.State.module_name = package + ".state"
        classes.State.class_name = "StateSW1L"

        classes.InitFields.module_name = package + ".init_fields"
        classes.InitFields.class_name = "InitFieldsSW1L"

        classes.Output.module_name = package + ".output"
        classes.Output.class_name = "OutputSW1L"

        classes.Forcing.module_name = package + ".forcing"
        classes.Forcing.class_name = "ForcingSW1L"


class Simul(SimulBasePseudoSpectral):
    """A solver of the shallow-water 1 layer equations (SW1L).

    .. inheritance-diagram:: Simul

    """

    InfoSolver = InfoSolverSW1L

    @staticmethod
    def _complete_params_with_default(params):
        r"""This static method is used to complete the *params* container.

        Notes
        -----

        - :math:`f` is the system rotation.

        - :math:`c` is the phase speed of surface gravity waves in the short-
          wave limit.

        - :math:`k_d` is the Rossby deformation wavenumber.

        - :math:`\beta` is differential rotation parameter, approximately given
          by :math:`\partial_y f`.

        The present solver will not work in the beta-plane. However a
        vorticity-divergence formulation can be solved with a beta term.

        """
        SimulBasePseudoSpectral._complete_params_with_default(params)

        attribs = {"f": 0, "c2": 20, "kd2": 0, "beta": 0}
        params._set_attribs(attribs)

    def __init__(self, params):
        # Parameter(s) specific to this solver

        params.f = float(params.f)
        params.c2 = float(params.c2)

        params.kd2 = params.f**2 / params.c2
        if params.beta != 0:
            raise NotImplementedError(
                "Do not use this solver for beta-plane! "
                "Equations are non-periodic in this formulation."
            )

        super().__init__(params)

        if mpi.rank == 0:
            self.output.print_stdout(
                "c2 = {:6.5g} ; f = {:6.5g} ; kd2 = {:6.5g}".format(
                    params.c2, params.f, params.kd2
                )
            )

    def tendencies_nonlin(self, state_spect=None, old=None):
        r"""Compute the nonlinear tendencies.

        Parameters
        ----------

        state_spect : :class:`fluidsim.base.setofvariables.SetOfVariables`
            optional

            Array containing the state, i.e. the horizontal velocities and
            surface displacement scalar in Fourier space.  When `state_spect` is
            provided, the variables vorticity and the velocities and surface
            displacement are computed from it, otherwise, they are taken from
            the global state of the simulation, `self.state`.

            These two possibilities are used during the Runge-Kutta
            time-stepping.

        Returns
        -------

        tendencies_fft : :class:`fluidsim.base.setofvariables.SetOfVariables`
            An array containing the tendencies for the velocities
            :math:`\mathbf u` and the surface displacement scalar :math:`\eta`.

        Notes
        -----

        .. |p| mathmacro:: \partial

        The one-layer shallow water equations can be written as:


        .. math:: \partial_t \mathbf u  = - \nabla (|\mathbf u|^2/2 + c^2 \eta)
                        - (\zeta + f) \times \mathbf u
        .. math:: \partial_t \eta = - \nabla. ((\eta + H) \mathbf u)

        This function computes all terms on the RHS of the equations above,
        including the nonlinear term. The linear dissipation term (not shown
        above) is computed implicitly, as demonstrated in
        :class:`fluidsim.base.time_stepping.pseudo_spect.TimeSteppingPseudoSpectral`.

        """
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
            tendencies_fft = SetOfVariables(
                like=self.state.state_spect, info="tendencies_nonlin"
            )
        else:
            tendencies_fft = old

        F1x, F1y = compute_Frot(rot, ux, uy, self.params.f)
        gradx_fft, grady_fft = oper.gradfft_from_fft(
            oper.fft2(compute_pressure(self.params.c2, eta, ux, uy))
        )
        oper.dealiasing(gradx_fft, grady_fft)

        Fx_fft = tendencies_fft.get_var("ux_fft")
        Fy_fft = tendencies_fft.get_var("uy_fft")
        Feta_fft = tendencies_fft.get_var("eta_fft")

        Fx_fft[:] = oper.fft2(F1x) - gradx_fft
        Fy_fft[:] = oper.fft2(F1y) - grady_fft

        Feta_fft[:] = -oper.divfft_from_vecfft(
            oper.fft2((eta + 1) * ux), oper.fft2((eta + 1) * uy)
        )

        oper.dealiasing(tendencies_fft)

        if self.params.forcing.enable:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


if __name__ == "__main__":

    import numpy as np

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    nh = 32
    Lh = 2 * np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    delta_x = params.oper.Lx / params.oper.nx
    params.nu_8 = (
        2.0 * 10e-1 * params.forcing.forcing_rate ** (1.0 / 3) * delta_x**8
    )

    params.time_stepping.t_end = 1.0
    # params.time_stepping.USE_CFL = False
    # params.time_stepping.deltat0 = 0.01

    params.init_fields.type = "noise"

    params.forcing.enable = True
    params.forcing.type = "waves"

    params.output.periods_print.print_stdout = 0.25

    params.output.periods_save.phys_fields = 0.5
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.periods_save.increments = 0.5
    params.output.periods_save.pdf = 0.5
    params.output.periods_save.time_signals_fft = True

    params.output.periods_plot.phys_fields = 0.0

    params.output.phys_fields.field_to_plot = "eta"

    sim = Simul(params)

    sim.output.phys_fields.plot(key_field="eta")
    sim.time_stepping.start()
    sim.output.phys_fields.plot(key_field="eta")

    fld.show()
