from fluidsim.util.testing import TestSimul
from fluidsim.solvers.waves2d.solver import Simul


class TestSimulBase(TestSimul):

    Simul = Simul

    @classmethod
    def init_params(cls):
        params = cls.params = cls.Simul.create_default_params()

        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"

        nh = 32
        lx = 2.0
        params.oper.nx = nh
        params.oper.ny = nh
        params.oper.Lx = lx
        params.oper.Ly = lx

        params.oper.coef_dealiasing = 2.0 / 3
        params.nu_8 = 2.0

        params.time_stepping.t_end = 0.5
        params.time_stepping.USE_CFL = False

        return params


class TestOutput(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()

        params.init_fields.type = "noise"

        params.output.periods_save.phys_fields = 0.5
        params.output.periods_plot.phys_fields = 0.1

        params.output.ONLINE_PLOT_OK = True
        params.output.phys_fields.field_to_plot = "f"
        params.output.periods_print.print_stdout = 0.1

        # save all outputs!
        periods = params.output.periods_save
        for key in periods._key_attribs:
            periods[key] = 0.2

    def test_output(self):
        sim = self.sim
        sim.time_stepping.start()
