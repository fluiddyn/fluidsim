#!/usr/bin/env python
import click
import numpy as np
import matplotlib.pyplot as plt


def solve(Simul, coef_dealiasing):
    params = Simul.create_default_params()
    params.output.sub_directory = "bench_skew_sym"

    params.output.periods_save.phys_fields = 1.0
    params.short_name_type_run = f"test_coef_dealias={coef_dealiasing:.2f}"
    params.oper.Lx = 2 * np.pi
    params.oper.nx = 128
    params.oper.coef_dealiasing = coef_dealiasing
    params.nu_2 = 0.0
    params.time_stepping.USE_CFL = False
    params.time_stepping.deltat0 = 1e-2
    params.time_stepping.t_end = 1.0
    params.time_stepping.type_time_scheme = "RK2"
    params.init_fields.type = "in_script"

    params.output.periods_print.print_stdout = 0.5

    sim = Simul(params)

    # Initialize
    x = sim.oper.x
    Lx = sim.oper.Lx

    u = np.sin(x)
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


@click.command()
@click.option("--solver", type=click.Choice(["conv", "skew_sym"]))
@click.option("--dealias/--no-dealias", type=bool, default=False)
def run(solver, dealias):
    if solver == "conv":
        from fluidsim.solvers.burgers1d.solver import Simul
    else:
        from fluidsim.solvers.burgers1d.skew_sym.solver import Simul

    solve(Simul, 2./3 if dealias else 1.0)


if __name__ == "__main__":
    run()
