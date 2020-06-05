from math import pi
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from fluiddyn.io.redirect_stdout import stdout_redirected
from fluidsim.solvers.nl1d.solver import Simul


# init field
k_init = 10
a = 0.7


class Resolution:
    def __init__(self, nx=32):

        # Create parameters
        params = Simul.create_default_params()

        # Simulation size
        nx = params.oper.nx = nx

        params.oper.Lx = 2 * pi
        params.oper.coef_dealiasing = 1

        params.time_stepping.USE_T_END = False
        params.time_stepping.USE_CFL = False

        params.output.HAS_TO_SAVE = False

        params.init_fields.type = "in_script"

        with stdout_redirected():
            sim = Simul(params)

        s_init = 1 + a * np.cos(k_init * sim.oper.x)

        params_big = deepcopy(params)
        params_big.oper.nx = 64 * 4
        oper_big = type(sim.oper)(params_big)
        s_init_big = 1 + a * np.cos(k_init * oper_big.x)

        def compute_exact_solution(time):
            s_exact = s_init_big / (1 + time * s_init_big)
            cond = oper_big.kx <= sim.oper.kx.max()
            return oper_big.fft(s_exact)[cond]

        self.params = params
        self.sim = sim
        self.compute_exact_solution = compute_exact_solution

        self.s_init = s_init

    def one_time_step(
        self,
        type_time_scheme,
        dt=1e-3,
        coef_dealiasing=0.66,
        verbose=False,
        plot_fig=False,
    ):

        params = self.params
        sim = self.sim

        sim.time_stepping.it = 0
        sim.time_stepping.t = 0.0

        sim.state.init_statephys_from(s=self.s_init.copy())
        sim.state.statespect_from_statephys()

        params.oper.coef_dealiasing = coef_dealiasing
        sim.oper = type(sim.oper)(params)

        params.time_stepping.it_end = 1
        params.time_stepping.deltat0 = dt
        params.time_stepping.type_time_scheme = type_time_scheme
        sim.time_stepping.init_from_params()

        with stdout_redirected():
            sim.time_stepping.main_loop()

        s_exact_fft = self.compute_exact_solution(dt)
        s_fft = sim.state.get_var("s_fft")

        # look for single alias
        k_alias = self.params.oper.nx - 2 * k_init

        if plot_fig:
            fig, ax = plt.subplots()

            def plot(x, y, *args, **kwargs):
                y = y.copy()
                y[y == 0] = 1e-15
                y = np.log10(abs(y))
                y[y < -15] = -15
                ax.plot(x, y, *args, **kwargs)

            plot(sim.oper.kx, s_fft, "o", label=f"num nx = {sim.oper.nx}")
            plot(
                sim.oper.kx,
                s_exact_fft,
                "go",
                markersize=12,
                fillstyle="none",
                label=f"exact",
            )

            ax.plot([k_alias] * 2, [-15, 0], "r:")

            ax.set_title(f"{type_time_scheme}, {coef_dealiasing = }, {dt = }")
            ax.set_xlabel(r"$k$")
            ax.set_ylabel(r"$\log(|\hat{s}|)$")

            fig.legend()

        # print(f"\nFirst alias from k_0 at k = {k_alias}")
        max_error = abs(s_fft - s_exact_fft).max()
        try:
            ratio_1st_peak = abs(s_fft[k_alias] / s_exact_fft[k_init])
        except IndexError:
            ratio_1st_peak = 0.0

        results = {
            "max_error": max_error,
            "ratio_1st_peak": ratio_1st_peak,
        }

        if verbose:
            print(
                f"\ntype_time_scheme = {params.time_stepping.type_time_scheme}, "
                f"{coef_dealiasing = }, {dt = :}"
            )
            print(f"{ratio_1st_peak = :.2g}")
            print(f"{max_error = :.2g}")
        return sim, results


if __name__ == "__main__":

    type_time_schemes = ["RK2", "RK2_phaseshift", "Euler", "Euler_phaseshift"]

    dt = 0.01

    resolution = Resolution(32)
    one_time_step = resolution.one_time_step

    # sim, results = one_time_step(
    #     "RK2_phaseshift", dt=dt, coef_dealiasing=1, plot_fig=True, verbose=1
    # )

    # sim, results = one_time_step(
    #     "RK2", dt=dt, coef_dealiasing=0.66, plot_fig=True, verbose=1
    # )

    sim, results = one_time_step(
        "RK4", dt=dt, coef_dealiasing=0.66, plot_fig=True, verbose=1
    )

    plt.show()
