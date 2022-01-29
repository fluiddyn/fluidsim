import numpy as np
import matplotlib.pyplot as plt

from phase_shifting import Resolution


schemes = [
    "RK4",
    "RK2",
    "RK2_phaseshift",
    "RK2_phaseshift_random",
    "RK2_phaseshift_exact",
    "Euler",
    "Euler_phaseshift",
    "Euler_phaseshift_random",
]

steps = np.logspace(-2.5, 0, 100)


def make_figures(coef_dealiasing=0.66, nx=32):

    resolution = Resolution(nx)
    one_time_step = resolution.one_time_step

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()

    figures = (fig0, fig1)
    axes = (ax0, ax1)

    max_ratio = 0.0

    for scheme in schemes:

        ratios = np.empty_like(steps)
        errors = np.empty_like(steps)

        for istep, step in enumerate(steps):
            sim, results = one_time_step(
                scheme, dt=step, coef_dealiasing=coef_dealiasing, verbose=0
            )
            ratios[istep] = results["ratio_1st_peak"]
            errors[istep] = results["max_error"]

        max_ratio = max(max_ratio, ratios.max())

        ax0.plot(steps, ratios, label=f"{scheme}")
        ax1.plot(steps, errors, label=f"{scheme}")

    ax0.set_ylabel("ratio first peak")

    ax = axes[1]
    ax.set_ylabel("max error")
    ax.set_xlabel("dt")

    x = np.linspace(10 ** (-2.5), 0.03, 2)
    ax.plot(x, 2e-1 * x**1, "-k")
    x = np.linspace(10 ** (-2.5), 0.1, 2)
    ax.plot(x, x**2, "-.k")
    x = np.linspace(0.04, 0.2, 2)
    ax.plot(x, x**3, "--k")
    x = np.linspace(0.1, 1, 2)
    ax.plot(x, 2e-1 * x**5, ":k")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    title = f"{coef_dealiasing = }, {nx = }"

    for fig in figures:
        fig.legend()
        fig.suptitle(title)

    if not max_ratio:
        plt.close(fig0)


make_figures(0.66)
make_figures(0.66, 256)
make_figures(1)
plt.show()
