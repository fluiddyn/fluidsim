"""Forcing (:mod:`fluidsim.solvers.ns3d.forcing.milestone`)
===========================================================

.. autoclass:: ForcingMilestone3D
   :members:

"""

from fluiddyn.util import mpi

from fluidsim.base.forcing.milestone import ForcingMilestone, PeriodicUniform


class ForcingMilestone3D(ForcingMilestone):
    ndim = 3

    def _init_operators(self, sim):
        lx = sim.params.oper.Lx
        ly = sim.params.oper.Ly

        if self.params_milestone.nx_max is None:
            self._is_using_coarse_oper = False
            self.oper_coarse = self.oper2d_big
        else:
            self._is_using_coarse_oper = True

            if mpi.rank == 0:
                from fluidsim.operators.operators2d import (
                    OperatorsPseudoSpectral2D,
                )

                params = OperatorsPseudoSpectral2D._create_default_params()
                params.oper.Lx = lx
                params.oper.Ly = ly
                nx = params.oper.nx = self.params_milestone.nx_max
                params.oper.ny = 2 * int(nx * ly / lx / 2)
                params.oper.type_fft = "sequential"
                self.oper_coarse = OperatorsPseudoSpectral2D(params)
                self.oper_coarse_shapeK_loc = self.oper_coarse.shapeK_loc
            else:
                self.oper_coarse = None
                self.oper_coarse_shapeK_loc = None

            if mpi.nb_proc > 1:
                self.oper_coarse_shapeK_loc = mpi.comm.bcast(
                    self.oper_coarse_shapeK_loc, root=0
                )

            self.solid = self.sim.oper.create_arrayX(value=0)
            self.solid_fft = self.sim.oper.create_arrayK(value=0)

    def __init__(self, sim):

        if mpi.rank == 0:
            from fluidsim.operators.operators2d import OperatorsPseudoSpectral2D

            params = OperatorsPseudoSpectral2D._create_default_params()
            params.oper.Lx = sim.params.oper.Lx
            params.oper.Ly = sim.params.oper.Ly
            params.oper.nx = sim.params.oper.nx
            params.oper.ny = sim.params.oper.ny
            params.oper.type_fft = "sequential"
            self.oper2d_big = OperatorsPseudoSpectral2D(params)
            self.solid2d_big_fft = self.oper2d_big.create_arrayK(value=0)
            self.solid2d_big = self.oper2d_big.create_arrayX(value=0)
        else:
            self.oper2d_big = "sequential"
            self.solid2d_big = None

        super().__init__(sim)

    def _full_from_coarse(self, solid):

        if self._is_using_coarse_oper:
            if mpi.rank == 0:
                solid_coarse_fft = self.oper_coarse.fft(solid)
                nKyc, nKxc = self.oper_coarse_shapeK_loc
                solid_coarse_fft[nKyc // 2, :] = 0.0
                solid_coarse_fft[:, nKxc - 1] = 0.0

                self.oper2d_big.put_coarse_array_in_array_fft(
                    solid_coarse_fft,
                    self.solid2d_big_fft,
                    self.oper_coarse,
                    self.oper_coarse_shapeK_loc,
                )
                self.oper2d_big.ifft_as_arg(
                    self.solid2d_big_fft, self.solid2d_big
                )
            else:
                solid_coarse_fft = None

        else:
            self.solid2d_big = solid

        self.solid = self.sim.oper.build_invariant_arrayX_from_2d_indices12X(
            self.solid2d_big, self.oper2d_big
        )

        return self.solid

    def compute(self, time=None):

        sim = self.sim

        if time is None:
            time = sim.time_stepping.t

        solid, x_coors, y_coors = self.get_solid_field(time)

        solid = self._full_from_coarse(solid)

        vx = sim.state.state_phys.get_var("vx")
        fx = self.sigma * solid * (self.get_speed(time) - vx)
        fx_fft = sim.oper.fft(fx)

        fy = -self.sigma * solid * sim.state.state_phys.get_var("vy")
        fy_fft = sim.oper.fft(fy)

        if sim.params.oper.NO_SHEAR_MODES:
            sim.oper.dealiasing(fx_fft, fy_fft)

        self.fstate.init_statespect_from(vx_fft=fx_fft, vy_fft=fy_fft)


if __name__ == "__main__":

    from time import perf_counter

    import matplotlib.pyplot as plt

    from fluidsim.solvers.ns3d.solver import Simul

    params = Simul.create_default_params()

    diameter = 0.5
    number_cylinders = 3
    speed = 0.1

    ny = params.oper.ny = 96 // 2
    nx = params.oper.nx = 4 * ny / 3

    ly = params.oper.Ly = 3 * diameter * number_cylinders
    lx = params.oper.Lx = ly / ny * nx * 4 / number_cylinders

    params.oper.nz = 8
    params.oper.Lz = 1.0

    params.forcing.enable = True
    params.forcing.type = "milestone"
    params.forcing.milestone.nx_max = 48
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

    params.time_stepping.t_end = movement.period * 1

    params.nu_8 = 1e-14

    params.output.sub_directory = "milestone"
    # params.output.periods_plot.phys_fields = 5.0
    params.output.periods_print.print_stdout = 10
    params.output.periods_save.phys_fields = 10.0

    sim = Simul(params)

    if mpi.rank == 0 and params.output.periods_plot.phys_fields:
        fig = plt.gcf()
        ax = fig.axes[0]
        ax.axis("equal")
        fig.set_size_inches(14, 4)

    self = milestone = sim.forcing.forcing_maker

    t_start = perf_counter()
    sim.time_stepping.start()
    mpi.printby0(f"Done in {perf_counter() - t_start} s")

    mpi.printby0(
        "For a video, run something like:\n\n"
        f"cd {sim.output.path_run};"
        'ipython -i -c "from fluidsim import load_sim_for_plot as load; '
        "sim=load(); sim.output.phys_fields.animate('vx')\""
    )
