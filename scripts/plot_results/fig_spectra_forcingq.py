#!/usr/bin/env python
# coding=utf8

import numpy as np
import matplotlib.pyplot as plt

from solveq2d import solveq2d

from create_figs_articles import CreateFigArticles


num_fig = 1000
SAVE_FIG = 1
name_file = "spectra_forcingq.eps"

create_fig = CreateFigArticles(
    short_name_article="SW1l", SAVE_FIG=SAVE_FIG, FOR_BEAMER=False, fontsize=19
)


name_dir_results = (
    create_fig.path_base_dir + "/Results_for_article_SW1l"
    "/Approach_runs_2048x2048"
    "/SE2D_SW1lexlin_forcing_L=50.x50._2048x2048_c2=400_f=0_2013-05-29_23-54-57"
)

sim = solveq2d.create_sim_plot_from_dir(name_dir_results)
tmin = 30
dict_results = sim.output.spectra.load2D_mean(tmin=tmin)

kh = dict_results["kh"]

EK = dict_results["spectrum2D_EK"]
EA = dict_results["spectrum2D_EA"]
EKr = dict_results["spectrum2D_EKr"]
E_tot = EK + EA
EKd = EK - EKr
Edlin = dict_results["spectrum2D_Edlin"]

fig, ax1 = create_fig.figure_axe(name_file=name_file)
ax1.set_xscale("log")
ax1.set_yscale("log")

coef_compensate = 5.0 / 3
coef_norm = kh**coef_compensate

l_Etot = ax1.plot(kh, E_tot * coef_norm, "k", linewidth=4)
l_EK = ax1.plot(kh, EK * coef_norm, "r", linewidth=2)
l_EA = ax1.plot(kh, EA * coef_norm, "b", linewidth=2)
ax1.plot(kh, EKr * coef_norm, "r--", linewidth=2)
ax1.plot(kh, EKd * coef_norm, "r:", linewidth=2)

ax1.plot(kh, -EK * coef_norm, "m", linewidth=2)
ax1.plot(kh, -EKd * coef_norm, "m:", linewidth=2)

ax1.plot(kh, Edlin * coef_norm, "y:", linewidth=2)


cond = np.logical_and(kh > 1, kh < 20)
ax1.plot(kh[cond], 1e1 * kh[cond] ** (-3.0) * coef_norm[cond], "k--", linewidth=1)
plt.figtext(0.6, 0.78, "$k^{-3}$", fontsize=20)

cond = np.logical_and(kh > 0.3, kh < 10)
ax1.plot(kh[cond], 4e-2 * kh[cond] ** (-2.0) * coef_norm[cond], "k:", linewidth=1)
plt.figtext(0.25, 0.55, "$k^{-2}$", fontsize=20)

cond = np.logical_and(kh > 0.3, kh < 15)
ax1.plot(
    kh[cond], 1e-2 * kh[cond] ** (-5.0 / 3) * coef_norm[cond], "k-.", linewidth=1
)
plt.figtext(0.5, 0.45, "$k^{-5/3}$", fontsize=20)


ax1.set_xlabel("$k_h$")
ax1.set_ylabel("2D spectra")


plt.rc("legend", numpoints=1)
leg1 = plt.figlegend(
    [l_Etot[0], l_EK[0], l_EA[0]],
    ["$E$", "$E_K$", "$E_A$"],
    loc=(0.78, 0.7),
    labelspacing=0.2,
)


ax1.set_xlim([0.1, 150])
ax1.set_ylim([1e-5, 2e1])


create_fig.save_fig()

plt.show()
