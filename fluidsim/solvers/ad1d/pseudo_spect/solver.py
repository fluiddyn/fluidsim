"""AD1D pseudospectral solver (:mod:`fluidsim.solvers.ad1d.solver`)
===================================================================


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


class InfoSolverAD1DPseudoSpect(InfoSolverPseudoSpectral):
    def _init_root(self):
        super()._init_root()

        package_base = "fluidsim.solvers.ad1d"
        package = package_base + ".pseudo_spect"
        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "AD1D"

        classes = self.classes

        classes.Operators.module_name = "fluidsim.operators.operators1d"
        classes.Operators.class_name = "OperatorsPseudoSpectral1D"

        classes.InitFields.module_name = package_base + ".init_fields"
        classes.InitFields.class_name = "InitFieldsAD1D"

        classes.State.module_name = package + ".state"
        classes.State.class_name = "StateAD1DPseudoSpectral"

        classes.Output.module_name = "fluidsim.solvers.ad1d.output"
        classes.Output.class_name = "Output"


class Simul(SimulBasePseudoSpectral):
    InfoSolver = InfoSolverAD1DPseudoSpect

    @staticmethod
    def _complete_params_with_default(params):
        SimulBasePseudoSpectral._complete_params_with_default(params)
        params._set_attrib("U", 1.0)

    def tendencies_nonlin(self, state_spect=None, old=None):

        if old is None:
            tendencies_fft = SetOfVariables(
                like=self.state.state_spect, info="tendencies_nonlin"
            )
        else:
            tendencies_fft = old

        tendencies_fft[:] = 0.0

        return tendencies_fft

    def compute_freq_complex(self, key):
        assert key == "s_fft"
        omega = 1.0j * self.params.U * self.oper.KX
        return omega


if __name__ == "__main__":
    from scipy.signal import gausspulse

    params = Simul.create_default_params()
    params.output.sub_directory = "examples"

    params.output.periods_save.phys_fields = 1.0
    params.short_name_type_run = "test"
    params.oper.Lx = 10
    params.oper.nx = 256
    params.nu_2 = 1e-3
    params.time_stepping.USE_CFL = False
    params.time_stepping.deltat0 = 1e-3
    params.time_stepping.t_end = 1.0
    params.time_stepping.type_time_scheme = "RK2"
    params.init_fields.type = "in_script"

    sim = Simul(params)

    # Initialize
    x = sim.oper.x
    Lx = sim.oper.Lx
    s = gausspulse(x - Lx / 2, fc=2)
    s_fft = sim.oper.fft(s)

    sim.state.init_statephys_from(s=s)
    sim.state.init_statespect_from(s_fft=s_fft)

    # Plot once
    # sim.output.init_with_initialized_state()
    # sim.output.phys_fields.plot(field='s')

    sim.time_stepping.start()
    # sim.output.phys_fields.plot(field='s')
