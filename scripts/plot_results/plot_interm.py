import matplotlib.pylab as plt
import glob
import numpy as np

from solveq2d import solveq2d


from create_figs_articles import CreateFigArticles


SAVE_FIG = False

create_fig = CreateFigArticles(
    short_name_article="SW1l", SAVE_FIG=SAVE_FIG, FOR_BEAMER=False, fontsize=19
)

dir_base = create_fig.path_base_dir + "/Results_SW1lw"

c = 40

resol = 240 * 2**5

str_resol = repr(resol)
str_to_find_path = (
    dir_base + "/Pure_standing_waves_" + str_resol + "*/SE2D*c=" + repr(c)
) + "_*"
print(str_to_find_path)

paths_dir = glob.glob(str_to_find_path)

print(paths_dir)


sim = solveq2d.create_sim_plot_from_dir(paths_dir[0])

tmin = sim.output.spatial_means.first_saved_time()
tstatio = tmin + 4.0


tmax = 1000


key_var = "uy"

(
    pdf_timemean,
    values_inc_timemean,
    nb_rx_to_plot,
) = sim.output.increments.load_pdf_from_file(
    tmin=tmin, tmax=tmax, key_var=key_var
)


deltax = sim.param.Lx / sim.param.nx
rxs = np.array(sim.output.increments.rxs, dtype=np.float64) * deltax


# if 7680
rmin = 8 * deltax
rmax = 40 * deltax

# rmin = 0.6
# rmax = 3


# if 4096
# rmin = 8e-2
# rmin = 8*deltax
# rmax = 5e-1
# rmax = 50*deltax

# if 2048
# rmin = 1.5e-1
# rmin = 5*deltax
# rmax = 20*deltax
# rmax = 3e-1


condr = np.logical_and(rxs > rmin, rxs < rmax)


def expo_from_order(order, PLOT=False, PLOT_PDF=False):
    order = float(order)
    M_order = np.empty(rxs.shape)
    for irx in xrange(rxs.size):
        deltainc = values_inc_timemean[irx, 1] - values_inc_timemean[irx, 0]
        M_order[irx] = deltainc * np.sum(
            pdf_timemean[irx] * abs(values_inc_timemean[irx]) ** order
        )

        # M_order[irx] = np.abs(deltainc*np.sum(
        #     pdf_timemean[irx]
        #     *values_inc_timemean[irx]**order
        #     ))

    pol = np.polyfit(np.log(rxs[condr]), np.log(M_order[condr]), 1)
    expo = pol[0]

    print(f"order = {order:.2f} ; expo = {expo:.2f}")
    M_lin = np.exp((pol[1] + np.log(rxs[condr]) * pol[0]))

    if PLOT:
        fig, ax1 = sim.output.figure_axe()
        title = (
            "struct. function, solver "
            + sim.output.name_solver
            + f", nh = {sim.param.nx:5d}"
            + ", c = {:.4g}, f = {:.4g}".format(
                np.sqrt(sim.param.c2), sim.param.f
            )
        )

        ax1.set_xlabel("$r$")
        ax1.set_ylabel(r"$\langle |\delta v|^{" + f"{order:.2f}" + "}\\rangle/r$")

        ax1.set_title(title)
        ax1.hold(True)
        ax1.set_xscale("log")
        ax1.set_yscale("log")

        norm = rxs ** (1)

        ax1.plot(rxs, M_order / norm, "x-", linewidth=1)

        ax1.plot(rxs[condr], M_order[condr] / norm[condr], "x-r", linewidth=1)

        ax1.plot(rxs[condr], M_lin / norm[condr], "y", linewidth=2)

        l_1 = ax1.plot(rxs, rxs / norm, "k", linewidth=1)
        l_K41 = ax1.plot(rxs, rxs ** (order / 3) / norm, "k--", linewidth=1)

        cond = rxs < 4 * deltax
        temp = rxs ** (order) / norm
        l_smooth = ax1.plot(
            rxs[cond],
            temp[cond] / temp[0] * M_order[0] / norm[0],
            "k:",
            linewidth=1,
        )

        plt.text(1, 6 * 10**0, f"order = {order:.2f}", fontsize=16)
        plt.text(0.11, 4 * 10**-1, f"expo = {expo:.3f}", fontsize=16)

        plt.rc("legend", numpoints=1)
        leg1 = ax1.legend(
            [l_smooth[0], l_K41[0], l_1[0]],
            ["smooth $r^q$", "K41 $r^{q/3}$", "shocks $r^1$"],
            loc=0
            # labelspacing = 0.2
        )

    if PLOT_PDF:
        fig, ax1 = sim.output.figure_axe()
        title = (
            "pdf increments, solver "
            + sim.output.name_solver
            + f", nh = {sim.param.nx:5d}"
            + f", c2 = {sim.param.c2:.4g}, f = {sim.param.f:.4g}"
        )

        ax1.set_title(title)
        ax1.hold(True)
        ax1.set_xscale("linear")
        ax1.set_yscale("linear")

        ax1.set_xlabel(key_var)
        ax1.set_ylabel(r"PDF x $\delta v^" + repr(order) + "$")

        colors = ["k", "y", "r", "b", "g", "m", "c"]

        irx_to_plot = [10, 50, 100]
        for irxp, irx in enumerate(irx_to_plot):

            val_inc = values_inc_timemean[irx]

            ax1.plot(
                val_inc,
                pdf_timemean[irx] * abs(val_inc) ** order,
                colors[irxp] + "x-",
                linewidth=1,
            )

    return expo


PLOT = False
PLOT = True
if PLOT:
    delta_order = 1
else:
    delta_order = 0.05


orders = np.arange(0.0, 6, delta_order)
expos = np.empty(orders.shape)

for iorder, order in enumerate(orders):
    expos[iorder] = expo_from_order(order, PLOT=PLOT, PLOT_PDF=False)

expos_K41 = orders / 3


fig, ax1 = sim.output.figure_axe()
title = (
    "intermittency, solver "
    + sim.output.name_solver
    + f", nh = {sim.param.nx:5d}"
    + ", c = {:.4g}, f = {:.4g}".format(np.sqrt(sim.param.c2), sim.param.f)
)

ax1.set_title(title)
ax1.hold(True)
ax1.set_xscale("linear")
ax1.set_yscale("linear")

ax1.set_xlabel("order q")
ax1.set_ylabel(r"$\zeta_q$")

ax1.plot(orders, expos)

ax1.plot(orders, expos_K41, "k--")

solveq2d.show()
