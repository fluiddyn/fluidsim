"""

run -m fluidsim.base.forcing.milestone

solid_coarse -> solid

forcing(x, y, z) = sigma * solid(x, y) * (speed_target - velocity(x, y, z))

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fluidsim.operators.operators2d import OperatorsPseudoSpectral2D
from .specific import SpecificForcingPseudoSpectralSimple as Base


def step(x, limit, smoothness):
    return 0.5 * (np.tanh((-x + limit) / smoothness) + 1)


class ForcingMilestone(Base):
    tag = "milestone"

    @classmethod
    def _complete_params_with_default(cls, params):
        Base._complete_params_with_default(params)
        milestone = params.forcing._set_child(cls.tag)
        milestone._set_child(
            "objects", dict(type="cylinders", number=2, diameter=1.0,),
        )
        movement = milestone._set_child("movement", dict(type="uniform"))
        movement._set_child("uniform", dict(speed=1.0))
        movement._set_child("sinusoidal", dict(length=1.0, period=1.0))
        movement._set_child(
            "periodic_uniform", dict(length=1.0, lenght_acc=1.0, speed=1.0)
        )

    def __init__(self, sim):
        super().__init__(sim)

        # params = OperatorsPseudoSpectral2D._create_default_params()
        # lx = params.oper.Lx = sim.params.oper.Lx
        # ly = params.oper.Ly = sim.params.oper.Ly
        # nx = 40
        # ny = int(nx * ly / lx)
        # params.oper.nx = nx
        # params.oper.ny = ny
        # self.oper_coarse = OperatorsPseudoSpectral2D(params)

        # for now, let's keep it simple!
        self.oper_coarse = sim.oper

        self.params_milestone = sim.params.forcing.milestone

        self.number_objects = self.params_milestone.objects.number
        mesh = self.oper_coarse.ly / self.number_objects

        self.speed = self.params_milestone.movement.uniform.speed

        self.y_coors = mesh * (1 / 2 + np.arange(self.number_objects))

        # calculus of coef_sigma

        # f(t) = f0 * exp(-sigma*t)
        # If we want f(t)/f0 = 10**(-gamma) after n_dt time steps, we have to have:
        # sigma = gamma / (n_dt * dt)

        gamma = 2
        n_dt = 4
        self.coef_sigma = gamma / n_dt

    def get_solid_field(self, time):

        oper = self.oper_coarse
        lx = oper.Lx
        radius = self.params_milestone.objects.diameter / 2

        x_coors, y_coors = self.get_locations(time)
        solid = np.zeros_like(oper.X)
        for x, y in zip(x_coors, y_coors):
            for index_x_periodicity in range(-1, 2):
                x_center = x + index_x_periodicity * lx
                distance_from_center = np.sqrt(
                    (oper.X - x_center) ** 2 + (oper.Y - y) ** 2
                )
                solid += step(distance_from_center, radius, 0.2)
        return solid, x_coors, y_coors

    def get_locations(self, time):
        speed = self.params_milestone.movement.uniform.speed
        lx = self.params.oper.Lx
        x_coors = (speed * time) % lx * np.ones(self.number_objects)
        return x_coors, self.y_coors

    def check_plot_solid(self, time):

        solid, x_coors, y_coors = self.get_solid_field(time)

        fig, ax = plt.subplots()
        oper_c = self.oper_coarse
        lx = oper_c.Lx
        ly = oper_c.Ly

        nx = oper_c.nx
        ny = oper_c.ny
        xs = np.linspace(0, lx, nx + 1)
        ys = np.linspace(0, ly, ny + 1)
        pcmesh = ax.pcolormesh(xs, ys, solid)
        ax.axis("equal")
        ax.set_xlim((0, lx))
        ax.set_ylim((0, ly))
        fig.colorbar(pcmesh)

    def check_plot_forcing(self, time):

        self.compute(time)

        fx = self.fstate.state_phys.get_var("ux")
        fy = self.fstate.state_phys.get_var("uy")

        rot_f = self.fstate.state_phys.get_var("rot")

        fig, ax = plt.subplots()
        oper_c = self.oper_coarse
        lx = oper_c.Lx
        ly = oper_c.Ly

        nx = oper_c.nx
        ny = oper_c.ny
        xs = np.linspace(0, lx, nx + 1)
        ys = np.linspace(0, ly, ny + 1)
        pcmesh = ax.pcolormesh(xs, ys, rot_f)

        ax.quiver(oper_c.X, oper_c.Y, fx, fy)

        ax.axis("equal")
        ax.set_xlim((0, lx))
        ax.set_ylim((0, ly))
        fig.colorbar(pcmesh)

    def check_with_animation(self):

        oper_c = self.oper_coarse

        lx = oper_c.Lx
        ly = oper_c.Ly

        nx = oper_c.nx
        ny = oper_c.ny

        movement = self.params_milestone.movement
        type_movement = movement.type

        if type_movement == "uniform":
            period = lx / movement.uniform.speed
            number_frames = 40
            dt = period / number_frames
        else:
            raise NotImplementedError

        fig, ax = plt.subplots()

        solid, x_coors, y_coors = self.get_solid_field(0)

        scat = ax.scatter(x_coors, y_coors, marker="x")
        ax.axis("equal")

        xs = np.linspace(0, lx, nx + 1)
        ys = np.linspace(0, ly, ny + 1)
        pcmesh = ax.pcolormesh(xs, ys, solid)

        ax.set_xlim((0, lx))
        ax.set_ylim((0, ly))
        fig.colorbar(pcmesh)

        def update_plot(index_time, pcmesh, scat):
            time = index_time * dt
            solid, x_coors, y_coors = self.get_solid_field(time)
            scat.set_offsets(np.vstack((x_coors, y_coors)).T)
            pcmesh.set_array(solid.flatten())
            return (pcmesh, scat)

        animation.FuncAnimation(
            fig,
            update_plot,
            frames=range(number_frames),
            fargs=(pcmesh, scat),
            blit=True,
            interval=500,  # in ms
        )
        plt.show()

    def compute(self, time=None):

        if time is None:
            time = self.sim.time_stepping.t

        solid, x_coors, y_coors = self.get_solid_field(time)

        ux = self.sim.state.state_phys.get_var("ux")
        uy = self.sim.state.state_phys.get_var("uy")

        fx = self.coef_sigma * solid * (self.speed - ux)
        fy = -self.coef_sigma * solid * uy

        fx_fft = self.sim.oper.fft(fx)
        fy_fft = self.sim.oper.fft(fy)

        # fy_fft = self.sim.oper.create_arrayK(value=0)

        # rot_fft = self.oper.rotfft_from_vecfft(fx_fft, fy_fft)
        # self.fstate.init_statespect_from(rot_fft=rot_fft)

        self.fstate.init_statespect_from(ux_fft=fx_fft, uy_fft=fy_fft)


if __name__ == "__main__":

    from fluidsim.solvers.ns2d.with_uxuy import Simul

    params = Simul.create_default_params()

    nx = params.oper.nx = 512
    ny = params.oper.ny = nx // 4

    lx = params.oper.Lx = 40.0
    params.oper.Ly = lx / nx * ny

    params.time_stepping.t_end = 100

    params.forcing.enable = True
    params.forcing.type = "milestone"
    params.forcing.milestone.objects.number = 2
    params.forcing.milestone.movement.uniform.speed = 1.0

    params.init_fields.type = "noise"
    params.init_fields.noise.velo_max = 1e-2

    params.nu_8 = 1e-10

    params.output.periods_plot.phys_fields = 2e-1
    params.output.periods_print.print_stdout = 5

    sim = Simul(params)

    fig = plt.gcf()
    ax = fig.axes[0]
    ax.axis("equal")
    fig.set_size_inches(14, 4)

    self = milestone = sim.forcing.forcing_maker

    sim.time_stepping.start()

    # milestone.check_with_animation()
    # milestone.check_plot_solid(0)
    # self.check_plot_forcing(10)
