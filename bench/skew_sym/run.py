#!/usr/bin/env python
"""Simulate Burgers equation with a sine-wave as an initial condition.

Notes
-----
- A stark difference in the final energy is observed between the convective and
  skew symmetric Burgers solver is evident a nx=101 is chosen.
- This contrast is less evident if nx=64, 128 etc. which is indeed puzzling.
- As suggested in the reference below the foolproof solution is to apply
  dealiasing which guarantees energy conservation.
- The time scheme also plays a role in influencing the final energy, but is
  less (?) likely to affect aliasing errors directly.
- It seems that the oddball mode was not set to zero in FluidSim, but this is
  perhaps not a big problem since we always use some dealiasing?

Examples
--------

.. seealso: https://kth-nek5000.github.io/kthNekBook/_notebooks/burgers.html

"""
import click
import numpy as np
import matplotlib.pyplot as plt


def solve(Simul, nx, time_scheme, coef_dealiasing):
    params = Simul.create_default_params()
    params.output.sub_directory = "bench_skew_sym"

    params.output.periods_save.phys_fields = 1.0
    params.short_name_type_run = f"test_coef_dealias={coef_dealiasing:.2f}"
    params.oper.Lx = 2 * np.pi
    params.oper.nx = nx
    params.oper.coef_dealiasing = coef_dealiasing
    params.nu_2 = 0.0
    params.time_stepping.USE_CFL = False
    params.time_stepping.deltat0 = 1e-2
    params.time_stepping.t_end = 1.1
    params.time_stepping.type_time_scheme = time_scheme
    params.init_fields.type = "in_script"

    params.output.periods_print.print_stdout = 0.1

    sim = Simul(params)

    # Initialize
    x = sim.oper.x
    Lx = sim.oper.Lx

    u = np.sin(x)
    u_fft = sim.oper.fft(u)
    # Set "oddball mode" to zero
    u_fft[sim.oper.nkx - 1] = 0.0

    sim.state.init_statephys_from(u=u)
    sim.state.init_statespect_from(u_fft=u_fft)

    E_initial = sim.output.compute_energy()

    # Plot once
    sim.output.init_with_initialized_state()
    sim.output.phys_fields.plot(field="u")
    ax = plt.gca()
    ax.grid()
    ax.set(xlim=(0, Lx))

    sim.time_stepping.start()

    ax.plot(x, sim.state.state_phys.get_var("u"))

    E_final = sim.output.compute_energy()
    print("Final Energy / Initial Energy =", E_final / E_initial)
    sim.output.print_stdout.plot_energy()
    ax2 = plt.gca()
    ax2.set(yscale="log")

    plt.show()


@click.command()
@click.option("--solver", type=click.Choice(["conv", "skew_sym"]))
@click.option("--nx", type=int, default=128)
@click.option("--ts", type=click.Choice(["Euler", "RK2", "RK4"]), default="Euler")
@click.option("--dealias/--no-dealias", type=bool, default=False)
def run(solver, nx, ts, dealias):
    """Execute Burgers solvers. Compare the following runs:

    \b
    # No dealiasing
    >>> ./run.py --nx 128 --solver conv
    >>> ./run.py --nx 128 --solver skew_sym

    \b
    # With dealiasing
    >>> ./run.py --nx 192 --dealias --solver conv
    >>> ./run.py --nx 192 --dealias --solver skew_sym

    """
    if solver == "conv":
        from fluidsim.solvers.burgers1d.solver import Simul
    else:
        from fluidsim.solvers.burgers1d.skew_sym.solver import Simul

    solve(Simul, nx, ts, 2.0 / 3 if dealias else 1.0)


if __name__ == "__main__":
    params = {
        "figure.figsize": (14, 5),
        "grid.alpha": 0.6,
    }
    plt.rcParams.update(params)
    run()
