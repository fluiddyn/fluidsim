import numpy as np

from fluiddyn.util import mpi

from .with_uxuy import Simul
from .solver import Simul as SimulBase

from .test_solver import TestSimulBase as Base


class TestSimulBase(Base):
    Simul = Simul


class TestSolverNS2DTendency(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()

        params.output.HAS_TO_SAVE = False

    def test_tendency(self):
        sim = self.sim
        ux_fft = sim.state.get_var("ux_fft")
        uy_fft = sim.state.get_var("uy_fft")

        tend = sim.tendencies_nonlin(state_spect=sim.state.state_spect)
        Fx_fft = tend.get_var("ux_fft")
        Fy_fft = tend.get_var("uy_fft")

        T_rot = np.real(Fx_fft.conj() * ux_fft + Fy_fft.conj() * uy_fft)

        ratio = sim.oper.sum_wavenumbers(T_rot) / sim.oper.sum_wavenumbers(
            abs(T_rot)
        )

        self.assertGreater(1e-15, abs(ratio))


class TestForcingMilestone(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.forcing.enable = True
        params.forcing.type = "milestone"
        movement = params.forcing.milestone.movement
        movement.type = "uniform"
        movement.uniform.speed = 1.0
        return params

    def test_milestone(self):
        self.sim.time_stepping.start()


class TestForcingMilestoneSinusoidal(TestForcingMilestone):
    Simul = SimulBase

    @classmethod
    def init_params(self):
        params = super().init_params()
        movement = params.forcing.milestone.movement
        movement.type = "sinusoidal"
        movement.sinusoidal.length = 6.0
        movement.sinusoidal.period = 100.0


class TestForcingMilestonePeriodicUniform(TestForcingMilestone):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.oper.NO_SHEAR_MODES = True
        params.time_stepping.t_end = 2.0
        params.forcing.milestone.nx_max = 16
        movement = params.forcing.milestone.movement
        movement.type = "periodic_uniform"
        movement.periodic_uniform.length = 2.0
        movement.periodic_uniform.length_acc = 0.25
        movement.periodic_uniform.speed = 2.5

    def test_milestone(self):
        super().test_milestone()

        if mpi.nb_proc > 1:
            return

        milestone = self.sim.forcing.forcing_maker
        print(milestone.period)
        milestone.check_plot_forcing(8.0)
        milestone.check_plot_solid(8.0)
        milestone.check_with_animation(number_frames=4, interval=1)
