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

import sys

import numpy as np

from transonic import boost, Array

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral,
    InfoSolverPseudoSpectral,
)

Af = Array[np.float64, "2d"]


@boost
def compute_Frot(ux: Af, uy: Af, px_rot: Af, py_rot: Af, beta: float = 0):
    if beta == 0:
        return -ux * px_rot - uy * py_rot

    else:
        return -ux * px_rot - uy * (py_rot + beta)


class InfoSolverNS2D(InfoSolverPseudoSpectral):
    """Contain the information on the solver ns2d.

    .. inheritance-diagram:: InfoSolverNS2D

    """

    def _init_root(self):
        """Init. `self` by writing the information on the solver.

        The function `InfoSolverPseudoSpectral._init_root` is
        called. We keep two classes listed by this function:

        - :class:`fluidsim.base.time_stepping.pseudo_spect.TimeSteppingPseudoSpectral`

        - :class:`fluidsim.operators.operators2d.OperatorsPseudoSpectral2D`


        The other first-level classes for this solver are:

        - :class:`fluidsim.solvers.ns2d.solver.Simul`

        - :class:`fluidsim.solvers.ns2d.state.StateNS2D`

        - :class:`fluidsim.solvers.ns2d.init_fields.InitFieldsNS2D`

        - :class:`fluidsim.solvers.ns2d.output.Output`

        - :class:`fluidsim.solvers.ns2d.forcing.ForcingNS2D`

        """
        super()._init_root()

        package = "fluidsim.solvers.ns2d"
        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "NS2D"

        classes = self.classes

        classes.State.module_name = package + ".state"
        classes.State.class_name = "StateNS2D"

        classes.InitFields.module_name = package + ".init_fields"
        classes.InitFields.class_name = "InitFieldsNS2D"

        classes.Output.module_name = package + ".output"
        classes.Output.class_name = "Output"

        classes.Forcing.module_name = package + ".forcing"
        classes.Forcing.class_name = "ForcingNS2D"


class Simul(SimulBasePseudoSpectral):
    """Pseudo-spectral solver 2D incompressible Navier-Stokes equations.

    .. inheritance-diagram:: Simul

    """

    InfoSolver = InfoSolverNS2D

    @staticmethod
    def _complete_params_with_default(params):
        """Complete the `params` container (static method)."""
        SimulBasePseudoSpectral._complete_params_with_default(params)
        attribs = {"beta": 0.0}
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

        tendencies_fft : :class:`fluidsim.base.setofvariables.SetOfVariables`
            An array containing the tendencies for the vorticity.

        Notes
        -----

        .. |p| mathmacro:: \partial

        The 2D Navier-Stokes equation can be written as:

        .. math:: \p_t \hat\zeta = \hat N(\zeta) - \sigma(k) \hat \zeta,

        This function compute the nonlinear term ("tendencies")
        :math:`N(\zeta) = - \mathbf{u}\cdot \mathbf{\nabla} \zeta`.

        """
        # the operator and the fast Fourier transform
        oper = self.oper
        ifft_as_arg = oper.ifft_as_arg

        # get or compute rot_fft, ux and uy
        if state_spect is None:
            rot_fft = self.state.state_spect.get_var("rot_fft")
            ux = self.state.state_phys.get_var("ux")
            uy = self.state.state_phys.get_var("uy")
        else:
            rot_fft = state_spect.get_var("rot_fft")
            ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
            ux = self.state.field_tmp0
            uy = self.state.field_tmp1
            ifft_as_arg(ux_fft, ux)
            ifft_as_arg(uy_fft, uy)

        # "px" like $\partial_x$
        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)

        px_rot = self.state.field_tmp2
        py_rot = self.state.field_tmp3

        ifft_as_arg(px_rot_fft, px_rot)
        ifft_as_arg(py_rot_fft, py_rot)

        Frot = compute_Frot(ux, uy, px_rot, py_rot, self.params.beta)

        if old is None:
            tendencies_fft = SetOfVariables(like=self.state.state_spect)
        else:
            tendencies_fft = old

        Frot_fft = tendencies_fft.get_var("rot_fft")
        oper.fft_as_arg(Frot, Frot_fft)

        oper.dealiasing(Frot_fft)

        # import numpy as np
        # T_rot = np.real(Frot_fft.conj()*rot_fft + Frot_fft*rot_fft.conj())/2.
        # print(('sum(T_rot) = {0:9.4e} ; sum(abs(T_rot)) = {1:9.4e}'
        #       ).format(self.oper.sum_wavenumbers(T_rot),
        #                self.oper.sum_wavenumbers(abs(T_rot))))

        if self.params.forcing.enable:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


if "sphinx" in sys.modules:
    params = Simul.create_default_params()

    __doc__ += (
        "Default parameters\n"
        "------------------\n"
        ".. code-block:: xml\n\n    "
        + "\n    ".join(params.__str__().split("\n\n", 1)[1].split("\n"))
        + "\n"
        + params._get_formatted_docs()
    )


if __name__ == "__main__":

    from math import pi

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    params.oper.nx = params.oper.ny = nh = 32
    params.oper.Lx = params.oper.Ly = Lh = 2 * pi
    # params.oper.coef_dealiasing = 1.

    delta_x = Lh / nh

    params.nu_8 = (
        2.0 * 10e-1 * params.forcing.forcing_rate ** (1.0 / 3) * delta_x**8
    )

    params.time_stepping.t_end = 10.0

    params.init_fields.type = "dipole"

    params.forcing.enable = False
    params.forcing.type = "random"
    # 'Proportional'
    # params.forcing.type_normalize

    params.output.sub_directory = "tests"

    # params.output.periods_print.print_stdout = 0.25

    params.output.periods_save.phys_fields = 1.0
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spatial_means = 0.05
    # params.output.periods_save.spect_energy_budg = 0.5
    # params.output.periods_save.increments = 0.5

    # params.output.periods_plot.phys_fields = 2.0

    params.output.ONLINE_PLOT_OK = True

    # params.output.spectra.HAS_TO_PLOT_SAVED = True
    # params.output.spatial_means.HAS_TO_PLOT_SAVED = True
    # params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True
    # params.output.increments.HAS_TO_PLOT_SAVED = True

    params.output.phys_fields.field_to_plot = "rot"

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
