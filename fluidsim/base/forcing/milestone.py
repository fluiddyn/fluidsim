"""

`run -m fluidsim.base.forcing.milestone`

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

        params = OperatorsPseudoSpectral2D._create_default_params()
        lx = params.oper.Lx = sim.params.oper.Lx
        ly = params.oper.Ly = sim.params.oper.Ly
        nx = 40
        ny = int(nx * ly / lx)
        params.oper.nx = nx
        params.oper.ny = ny
        self.oper_coarse = OperatorsPseudoSpectral2D(params)

        self.params_milestone = sim.params.forcing.milestone

        self.number_objects = self.params_milestone.objects.number
        mesh = ly / self.number_objects

        self.y_coors = mesh * (1 / 2 + np.arange(self.number_objects))

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

        scat = ax.scatter(x_coors, y_coors)
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
        )
        plt.show()

    def compute(self):
        rot_fft = self.oper.create_arrayK(value=0.0)
        self.fstate.init_statespect_from(rot_fft=rot_fft)


if __name__ == "__main__":

    from fluidsim.solvers.ns2d.solver import Simul

    params = Simul.create_default_params()

    nx = params.oper.nx = 100
    ny = params.oper.ny = 40

    lx = params.oper.Lx = 10.0
    params.oper.Ly = lx / nx * ny

    params.forcing.enable = True
    params.forcing.type = "milestone"

    sim = Simul(params)

    milestone = sim.forcing.forcing_maker
    milestone.check_with_animation()
