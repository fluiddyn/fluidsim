"""Burgers1D solver (:mod:`fluidsim.solvers.burgers1d.solver`)
==============================================================

Provides:

.. autoclass:: Simul
   :members:
   :private-members:


"""
from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral,
    InfoSolverPseudoSpectral,
)
from fluidsim.base.state import StatePseudoSpectral


class StateBurgers1D(StatePseudoSpectral):
    """Contains the variables corresponding to the state and handles the
    access to other fields for the pseudospectral solver Burgers 1d.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": ["u_fft"],
                "keys_state_phys": ["u"],
                "keys_computable": [],
                "keys_phys_needed": ["u"],
            }
        )


class InfoSolver(InfoSolverPseudoSpectral):
    def _init_root(self):
        super()._init_root()

        package = "fluidsim.solvers.burgers1d"
        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "Burgers1D"

        classes = self.classes

        package_ad1d = "fluidsim.solvers.ad1d"

        classes.Operators.module_name = "fluidsim.operators.operators1d"
        classes.Operators.class_name = "OperatorsPseudoSpectral1D"

        classes.InitFields.module_name = package_ad1d + ".init_fields"
        classes.InitFields.class_name = "InitFieldsAD1D"

        classes.State.module_name = self.module_name
        classes.State.class_name = "StateBurgers1D"

        classes.Output.module_name = "fluidsim.solvers.ad1d.output"
        classes.Output.class_name = "Output"


class Simul(SimulBasePseudoSpectral):
    InfoSolver = InfoSolver

    def tendencies_nonlin(self, state_spect=None, old=None):

        if state_spect is None:
            u_fft = self.state.state_spect.get_var("u_fft")
            signal = self.state.state_phys.get_var("u")
        else:
            u_fft = state_spect.get_var("u_fft")
            signal = self.oper.ifft(u_fft)

        pxu_fft = self.oper.pxffft_from_fft(u_fft)
        pxu = self.oper.ifft(pxu_fft)

        f_signal = -signal * pxu

        if old is None:
            tendencies_fft = SetOfVariables(
                like=self.state.state_spect, info="tendencies_nonlin"
            )
        else:
            tendencies_fft = old

        f_fft = tendencies_fft.get_var("u_fft")
        self.oper.fft_as_arg(f_signal, f_fft)
        self.oper.dealiasing(f_fft)
        # Set "oddball mode" to zero
        f_fft[self.oper.nkx - 1] = 0.0
        return tendencies_fft


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    params = Simul.create_default_params()
    params.output.sub_directory = "examples"

    params.output.periods_save.phys_fields = 1.0
    params.short_name_type_run = "test"
    params.oper.Lx = 8
    params.oper.nx = 128
    params.oper.coef_dealiasing = 0.99
    params.nu_2 = 1e-2
    params.time_stepping.USE_CFL = False
    params.time_stepping.deltat0 = 1e-2
    params.time_stepping.t_end = 0.5
    params.time_stepping.type_time_scheme = "RK2"
    params.init_fields.type = "in_script"

    params.output.periods_print.print_stdout = 0.5

    sim = Simul(params)

    # Initialize
    x = sim.oper.x
    Lx = sim.oper.Lx

    u = np.zeros_like(x)

    coef = 4
    cond = x < Lx / 2
    u[cond] = np.tanh(coef * (x[cond] - Lx / 4))
    cond = x >= Lx / 2
    u[cond] = -np.tanh(coef * (x[cond] - 3 * Lx / 4))

    u_fft = sim.oper.fft(u)

    sim.state.init_statephys_from(u=u)
    sim.state.init_statespect_from(u_fft=u_fft)

    # Plot once
    sim.output.init_with_initialized_state()
    sim.output.phys_fields.plot(field="u")
    ax = plt.gca()

    sim.time_stepping.start()

    ax.plot(x, sim.state.state_phys.get_var("u"))
    plt.show()
