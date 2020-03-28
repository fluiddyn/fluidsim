"""

run -m fluidsim.base.forcing.milestone

solid_coarse -> solid

forcing(x, y, z) = sigma * solid(x, y) * (speed_target - velocity(x, y, z))

"""

from math import sin, cos, pi

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from fluidsim.operators.operators2d import OperatorsPseudoSpectral2D
from .specific import SpecificForcingPseudoSpectralSimple as Base


def step(x, limit, smoothness):
    return 0.5 * (np.tanh((-x + limit) / smoothness) + 1)


class PeriodicUniform:
    def __init__(self, speed_max, length, length_acc, lx):

        self.speed_max = speed_max

        sign = np.sign(speed_max)
        x_uni = length - 2 * length_acc
        t_uni = x_uni / abs(speed_max)
        self.acc = sign * speed_max ** 2 / (2 * length_acc)
        t_a = speed_max / self.acc
        self.period = 4 * t_a + 2 * t_uni

        self.t_1 = t_a
        self.t_2 = t_a + t_uni
        self.t_3 = self.t_2 + 2 * t_a
        self.t_4 = self.t_3 + t_uni

        self.x_0 = lx / 2 - sign * length / 2
        self.x_1 = self.x_4 = self.x_0 + sign * length_acc
        self.x_2 = self.x_3 = self.x_0 + sign * (length_acc + x_uni)

    def get_speed(self, time):

        speed_max = self.speed_max
        acc = self.acc
        t_1 = self.t_1
        t_2 = self.t_2
        t_3 = self.t_3
        t_4 = self.t_4

        time = time % self.period

        # acceleration
        if time <= t_1:
            return acc * time
        # uniform
        elif (time > t_1) and (time <= t_2):
            return speed_max
        # acceleration 2
        elif (time > t_2) and (time <= t_3):
            t = time - t_2
            return speed_max - acc * t
        # uniform 2
        elif (time > t_3) and (time <= t_4):
            return -speed_max
        # acceleration 3
        elif time > t_4:
            t = time - t_4
            return -speed_max + acc * t

    def get_locations(self, time):

        speed_max = self.speed_max
        acc = self.acc
        t_1 = self.t_1
        t_2 = self.t_2
        t_3 = self.t_3
        t_4 = self.t_4

        time = time % self.period

        # acceleration
        if time <= t_1:
            return self.x_0 + acc / 2 * time ** 2
        # uniform
        elif (time > t_1) and (time <= t_2):
            return self.x_1 + speed_max * (time - t_1)
        # acceleration 2
        elif (time > t_2) and (time <= t_3):
            t = time - t_2
            return self.x_2 + speed_max * t - acc / 2 * t ** 2
        # uniform 2
        elif (time > t_3) and (time <= t_4):
            return self.x_3 - speed_max * (time - t_3)
        # acceleration 3
        elif time > t_4:
            t = time - t_4
            return self.x_4 - speed_max * t + acc / 2 * t ** 2


class ForcingMilestone(Base):
    tag = "milestone"

    @classmethod
    def _complete_params_with_default(cls, params):
        Base._complete_params_with_default(params)
        milestone = params.forcing._set_child(cls.tag)
        milestone._set_child(
            "objects",
            dict(
                type="cylinders",
                number=2,
                diameter=1.0,
                width_boundary_layers=0.1,
            ),
        )
        movement = milestone._set_child("movement", dict(type="uniform"))
        movement._set_child("uniform", dict(speed=1.0))
        movement._set_child("sinusoidal", dict(length=1.0, period=1.0))
        movement._set_child(
            "periodic_uniform", dict(length=1.0, length_acc=1.0, speed=1.0)
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
        self.y_coors = mesh * (1 / 2 + np.arange(self.number_objects))

        # Calculus of coef_sigma:
        # f(t) = f0 * exp(-sigma*t)
        # If we want f(t)/f0 = 10**(-gamma) after n_dt time steps, we have to have:
        # sigma = gamma / (n_dt * dt)
        gamma = 2
        n_dt = 4
        self.coef_sigma = gamma / n_dt
        self.sigma = self.coef_sigma / self.params.time_stepping.deltat_max

        type_movement = self.params_milestone.movement.type
        if type_movement == "uniform":
            self.get_locations = self.get_locations_uniform
            self._speed = self.params_milestone.movement.uniform.speed
            self.get_speed = self.get_speed_uniform
        elif type_movement == "sinusoidal":
            sinusoidal = self.params_milestone.movement.sinusoidal
            self._half_length = sinusoidal.length / 2
            self._omega = 2 * pi / sinusoidal.period
            self.get_locations = self.get_locations_sinusoidal
            self.get_speed = self.get_speed_sinusoidal

        elif type_movement == "periodic_uniform":

            params_pu = self.params_milestone.movement.periodic_uniform
            periodic_uniform = PeriodicUniform(
                params_pu.speed,
                params_pu.length,
                params_pu.length_acc,
                sim.params.oper.Lx,
            )

            def get_locations(time):
                x_coors = periodic_uniform.get_locations(time) * np.ones(
                    self.number_objects
                )
                return x_coors, self.y_coors

            self.get_locations = get_locations
            self.get_speed = periodic_uniform.get_speed
            self.period = periodic_uniform.period
        else:
            raise NotImplementedError

    def get_solid_field(self, time):

        oper = self.oper_coarse
        lx = oper.Lx
        radius = self.params_milestone.objects.diameter / 2

        width = self.params_milestone.objects.width_boundary_layers

        x_coors, y_coors = self.get_locations(time)
        solid = np.zeros_like(oper.X)
        for x, y in zip(x_coors, y_coors):
            for index_x_periodicity in range(-1, 2):
                x_center = x + index_x_periodicity * lx
                distance_from_center = np.sqrt(
                    (oper.X - x_center) ** 2 + (oper.Y - y) ** 2
                )
                solid += step(distance_from_center, radius, width)
        return solid, x_coors, y_coors

    def get_locations_uniform(self, time):
        speed = self.params_milestone.movement.uniform.speed
        lx = self.params.oper.Lx
        x_coors = (speed * time) % lx * np.ones(self.number_objects)
        return x_coors, self.y_coors

    def get_speed_uniform(self, time):
        return self._speed

    def get_locations_sinusoidal(self, time):
        lx = self.params.oper.Lx
        x_coors = (
            lx / 2 + self._half_length * sin(self._omega * time - pi / 2)
        ) * np.ones(self.number_objects)
        return x_coors, self.y_coors

    def get_speed_sinusoidal(self, time):
        return self._half_length * self._omega * cos(self._omega * time - pi / 2)

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

        try:
            period = self.period
        except AttributeError:
            if type_movement == "uniform":
                period = lx / movement.uniform.speed
            elif type_movement == "sinusoidal":
                period = movement.sinusoidal.period
            else:
                raise NotImplementedError

        number_frames = 40
        dt = period / number_frames

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

        sim = self.sim

        if time is None:
            time = sim.time_stepping.t

        solid, x_coors, y_coors = self.get_solid_field(time)

        ux = sim.state.state_phys.get_var("ux")
        fx = self.sigma * solid * (self.get_speed(time) - ux)
        fx_fft = sim.oper.fft(fx)

        if "rot_fft" in sim.state.keys_state_spect:
            fy_fft = sim.oper.create_arrayK(value=0)
            rot_fft = self.oper.rotfft_from_vecfft(fx_fft, fy_fft)
            self.fstate.init_statespect_from(rot_fft=rot_fft)
        else:
            uy = sim.state.state_phys.get_var("uy")
            fy = -self.sigma * solid * uy
            fy_fft = sim.oper.fft(fy)
            self.fstate.init_statespect_from(ux_fft=fx_fft, uy_fft=fy_fft)


if __name__ == "__main__":

    from fluidsim.solvers.ns2d.with_uxuy import Simul

    # from fluidsim.solvers.ns2d.solver import Simul

    params = Simul.create_default_params()

    diameter = 0.5
    number_cylinders = 3
    speed = 0.1

    ny = params.oper.ny = 96 * 2
    nx = params.oper.nx = 4 * ny / 3

    ly = params.oper.Ly = 3 * diameter * number_cylinders
    lx = params.oper.Lx = ly / ny * nx * 4 / number_cylinders

    params.forcing.enable = True
    params.forcing.type = "milestone"
    objects = params.forcing.milestone.objects

    objects.number = number_cylinders
    objects.diameter = diameter
    objects.width_boundary_layers = 0.1

    movement = params.forcing.milestone.movement

    # movement.type = "uniform"
    # movement.uniform.speed = 1.0

    # movement.type = "sinusoidal"
    # movement.sinusoidal.length = 14 * diameter
    # movement.sinusoidal.period = 100.0

    movement.type = "periodic_uniform"
    movement.periodic_uniform.length = 14 * diameter
    movement.periodic_uniform.length_acc = 0.25
    movement.periodic_uniform.speed = speed

    params.init_fields.type = "noise"
    params.init_fields.noise.velo_max = 5e-3

    movement = PeriodicUniform(
        speed,
        movement.periodic_uniform.length,
        movement.periodic_uniform.length_acc,
        lx,
    )

    params.time_stepping.t_end = movement.period * 2

    params.nu_8 = 1e-14

    params.output.periods_plot.phys_fields = 5.0
    params.output.periods_print.print_stdout = 5

    sim = Simul(params)

    fig = plt.gcf()
    ax = fig.axes[0]
    ax.axis("equal")
    fig.set_size_inches(14, 4)

    self = milestone = sim.forcing.forcing_maker

    # sim.time_stepping.start()

    # milestone.check_with_animation()
    # milestone.check_plot_solid(0)
    # self.check_plot_forcing(10)
