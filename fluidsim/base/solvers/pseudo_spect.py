"""Base solver (:mod:`fluidsim.base.solvers.pseudo_spect`)
==========================================================

This module provides two base classes that can be used to define
pseudo-spectral solvers.

.. autoclass:: InfoSolverPseudoSpectral
   :members:
   :private-members:

.. autoclass:: SimulBasePseudoSpectral
   :members:
   :private-members:

.. autoclass:: InfoSolverPseudoSpectral3D
   :members:
   :private-members:

"""

import os

import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.base.solvers.base import SimulBase, InfoSolverBase


class InfoSolverPseudoSpectral(InfoSolverBase):
    """Contain the information on a base pseudo-spectral 2D solver."""

    def _init_root(self):
        """Init. `self` by writting the information on the solver.

        The first-level classes for this base solver are:

        - :class:`fluidsim.base.solvers.pseudo_spect.SimulBasePseudoSpectral`

        - :class:`fluidsim.base.state.StatePseudoSpectral`

        - :class:`fluidsim.base.time_stepping.pseudo_spect.TimeSteppingPseudoSpectral`

        - :class:`fluidsim.operators.operators.OperatorsPseudoSpectral2D`

        """
        super()._init_root()

        self.module_name = "fluidsim.base.solvers.pseudo_spect"
        self.class_name = "SimulBasePseudoSpectral"
        self.short_name = "BasePS"

        self.classes.State.module_name = "fluidsim.base.state"
        self.classes.State.class_name = "StatePseudoSpectral"

        self.classes.TimeStepping.module_name = (
            "fluidsim.base.time_stepping.pseudo_spect"
        )

        self.classes.TimeStepping.class_name = "TimeSteppingPseudoSpectral"

        self.classes.Operators.module_name = "fluidsim.operators.operators"
        if "FLUIDSIM_NO_FLUIDFFT" not in os.environ:
            self.classes.Operators.module_name += "2d"

        self.classes.Operators.class_name = "OperatorsPseudoSpectral2D"

        self.classes.Forcing.class_name = "ForcingBasePseudoSpectral"

        self.classes.Output.class_name = "OutputBasePseudoSpectral"

        self.classes._set_child(
            "Preprocess",
            attribs={
                "module_name": "fluidsim.base.preprocess.pseudo_spect",
                "class_name": "PreprocessPseudoSpectral",
            },
        )


class InfoSolverPseudoSpectral3D(InfoSolverPseudoSpectral):
    """Contain the information on a base pseudo-spectral 3D solver."""

    def _init_root(self):
        """Init. `self` by writting the information on the solver.

        The first-level classes for this base solver are the same as
        for the 2D pseudo-spectral base solver except the class:

        - :class:`fluidsim.operators.operators2d.OperatorsPseudoSpectral3D`

        """

        super()._init_root()

        self.classes.Operators.module_name = "fluidsim.operators.operators3d"
        self.classes.Operators.class_name = "OperatorsPseudoSpectral3D"


class SimulBasePseudoSpectral(SimulBase):
    """Pseudo-spectral base solver."""

    InfoSolver = InfoSolverPseudoSpectral

    @staticmethod
    def _complete_params_with_default(params):
        """Complete the `params` container (static method)."""
        SimulBase._complete_params_with_default(params)

        attribs = {"nu_8": 0.0, "nu_4": 0.0, "nu_m4": 0.0}
        params._set_attribs(attribs)

        params._set_doc(
            params._doc
            + """
nu_8: float

    Hyper-viscous coefficient of order 8. Also used in the method
    compute_freq_diss.

nu_4: float

    Hyper-viscous coefficient of order 4.

nu_m4: float

    Hypo-viscous coefficient of order -4. Hypo-viscosity affect more the large
    scales.

"""
        )

    def compute_freq_diss(self):
        r"""Compute the dissipation frequency.

        Use the `self.params.nu_...` parameters to compute an array
        containing the dissipation frequency as a function of the
        wavenumber.

        .. |p| mathmacro:: \partial

        The governing equations for a pseudo-spectral solver can be
        written as

        .. math:: \p_t S = N(S) - \sigma(k) S,

        where :math:`\sigma` is the frequency associated with the
        linear term.

        In this function, the frequency :math:`\sigma` is split in 2
        parts: the dissipation at small scales and the dissipation at
        large scale (hypo-viscosity, `params.nu_m4`).

        Returns
        -------

        f_d : `numpy.array`
            The dissipation frequency as a function of the wavenumber
            (small sclale part).

        f_d_hypo : `numpy.array`
            The dissipation frequency at large scale (hypo-viscosity)

        .. FIXME: Shouldn't fourth order viscosity be negative?

        """
        if self.params.nu_2 > 0:
            f_d = self.params.nu_2 * self.oper.K2
        else:
            f_d = np.zeros_like(self.oper.K2)

        if self.params.nu_4 > 0.0:
            f_d += self.params.nu_4 * self.oper.K4

        if self.params.nu_8 > 0.0:
            f_d += self.params.nu_8 * self.oper.K8

        if self.params.nu_m4 != 0.0:
            f_d_hypo = self.params.nu_m4 / self.oper.K2_not0 ** 2
            # mode K2 = 0 !
            if mpi.rank == 0:
                f_d_hypo[0, 0] = f_d_hypo[0, 1]

            f_d_hypo[self.oper.K <= 20] = 0.0

        else:
            f_d_hypo = np.zeros_like(f_d)

        return f_d, f_d_hypo

    def tendencies_nonlin(self, variables=None, old=None):
        r"""Compute the nonlinear tendencies.

        This function has to be overridden in a child class.

        Returns
        -------

        tendencies_fft : :class:`fluidsim.base.setofvariables.SetOfVariables`
            An array containing only zeros.

        """
        if old is None:
            tendencies = SetOfVariables(like=self.state.state_spect)
        else:
            tendencies = old

        tendencies.initialize(value=0.0)
        return tendencies


Simul = SimulBasePseudoSpectral


if __name__ == "__main__":

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    nh = 16
    Lh = 2 * np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    delta_x = params.oper.Lx / params.oper.nx
    params.nu_8 = (
        2.0 * 10e-1 * params.forcing.forcing_rate ** (1.0 / 3) * delta_x ** 8
    )

    params.time_stepping.t_end = 5.0

    params.init_fields.type = "noise"

    params.output.periods_plot.phys_fields = 0.0

    params.output.periods_print.print_stdout = 0.25
    params.output.periods_save.phys_fields = 2.0

    sim = Simul(params)

    sim.output.phys_fields.plot()
    sim.time_stepping.start()
    sim.output.phys_fields.plot()

    fld.show()
