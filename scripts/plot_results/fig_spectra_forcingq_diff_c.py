#!/usr/bin/env python
# coding=utf8

import numpy as np
import matplotlib.pyplot as plt

from solveq2d import solveq2d

from create_figs_articles import CreateFigArticles


num_fig = 1000
SAVE_FIG = 1
name_file = "spectra_forcingq_diff_c.eps"

create_fig = CreateFigArticles(
    short_name_article="SW1l", SAVE_FIG=SAVE_FIG, FOR_BEAMER=False, fontsize=19
)

dir_base = (
    create_fig.path_base_dir + "/Results_for_article_SW1l"
    "/Approach_runs_2048x2048"
)
set_of_dir_results = solveq2d.SetOfDirResults(dir_base=dir_base)
dirs = set_of_dir_results.dirs_from_values(solver="SW1lexlin", FORCING=True)

print(dirs)

tmin = 264


def sprectra_from_namedir(name_dir_results):
    path_dir_results = set_of_dir_results.path_dirs[name_dir_results]
    sim = solveq2d.create_sim_plot_from_dir(path_dir_results)
    dict_results = sim.output.spectra.load2D_mean(tmin=tmin)
    kh = dict_results["kh"]
    EK = dict_results["spectrum2D_EK"]
    EA = dict_results["spectrum2D_EA"]
    EKr = dict_results["spectrum2D_EKr"]
    EKd = EK - EKr
    return kh, EKr, EKd


kh, EKr0, EKd0 = sprectra_from_namedir(dirs[0])
kh, EKr1, EKd1 = sprectra_from_namedir(dirs[1])
kh, EKr2, EKd2 = sprectra_from_namedir(dirs[2])
kh, EKr3, EKd3 = sprectra_from_namedir(dirs[3])


fig, ax1 = create_fig.figure_axe(name_file=name_file)
ax1.set_xscale("log")
ax1.set_yscale("log")

coef_compensate = 5.0 / 3
coef_norm = kh**coef_compensate

ax1.plot(kh, EKr0 * coef_norm, "k--", linewidth=2)
ax1.plot(kh, EKd0 * coef_norm, "k:", linewidth=2)

ax1.plot(kh, EKr1 * coef_norm, "r--", linewidth=2)
ax1.plot(kh, EKd1 * coef_norm, "r:", linewidth=2)

ax1.plot(kh, EKr2 * coef_norm, "b--", linewidth=2)
ax1.plot(kh, EKd2 * coef_norm, "b:", linewidth=2)

ax1.plot(kh, EKr3 * coef_norm, "c--", linewidth=2)
ax1.plot(kh, EKd3 * coef_norm, "c:", linewidth=2)


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


# plt.rc('legend', numpoints=1)
# leg1 = plt.figlegend(
#         [l_Etot[0], l_EK[0], l_EA[0]],
#         ['$E$', '$E_K$', '$E_A$'],
#         loc=(0.78, 0.7),
#         labelspacing = 0.2
# )


ax1.set_xlim([0.1, 150])
ax1.set_ylim([1e-5, 2e1])


create_fig.save_fig()

plt.show()
