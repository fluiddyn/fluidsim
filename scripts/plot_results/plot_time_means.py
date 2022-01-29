#!/usr/bin/env python
# coding=utf8

import numpy as np
import glob
import os

import matplotlib.pylab as plt

from solveq2d import solveq2d

LOAD = True
# LOAD = False

dir_base = "Approach_runs_1024x1024"
# dir_base = 'Waves_statio_256x256'

set_of_dir_results = solveq2d.SetOfDirResults(dir_base=dir_base)

values_c2 = set_of_dir_results.values_c2
# values_c2 = [600]
values_solver = set_of_dir_results.values_solver
values_solver = ["SW1l"]

tstatio = 250


if LOAD:

    tuple_loop = [
        (c2, name_solver) for c2 in values_c2 for name_solver in values_solver
    ]

    nb_runs = 0
    for c2, name_solver in tuple_loop:
        nb_runs += 1

    arr_c2 = np.empty([nb_runs])
    Froude_numbers = np.empty([nb_runs])
    EK = np.empty([nb_runs])
    EA = np.empty([nb_runs])
    EKr = np.empty([nb_runs])
    epsK = np.empty([nb_runs])
    epsA = np.empty([nb_runs])
    epsK_tot = np.empty([nb_runs])
    epsA_tot = np.empty([nb_runs])

    irun = -1

    for c2, name_solver in tuple_loop:
        path_dir = set_of_dir_results.one_path_from_values(
            solver=name_solver, c2=c2
        )

        irun += 1
        sim = solveq2d.create_sim_plot_from_dir(name_dir=path_dir)

        (
            dict_time_means,
            dict_results,
        ) = sim.output.spatial_means.compute_time_means(tstatio)

        arr_c2[irun] = sim.param["c2"]
        EK[irun] = dict_time_means["EK"]

        Froude_numbers[irun] = 2 * EK[irun] / sim.param["c2"]

        EA[irun] = dict_time_means["EA"]
        EKr[irun] = dict_time_means["EKr"]
        epsK[irun] = dict_time_means["epsK"]
        epsA[irun] = dict_time_means["epsA"]
        epsK_tot[irun] = dict_time_means["epsK_tot"]
        epsA_tot[irun] = dict_time_means["epsA_tot"]

    eps = epsK + epsA
    eps_tot = epsK_tot + epsA_tot


# width_axe = 0.85
# height_axe = 0.37
# x_left_axe = 0.12
# z_bottom_axe = 0.56

# size_axe = [x_left_axe, z_bottom_axe,
#             width_axe, height_axe]
# fig, ax1 = sim.output.figure_axe(size_axe=size_axe)

fig, ax1 = sim.output.figure_axe()

ax1.set_xlabel("$2E/c^2$")
ax1.set_ylabel(r"$\epsilon / P$")
ax1.hold(True)

coef_norm = Froude_numbers**-1.54
ax1.loglog(Froude_numbers, Froude_numbers**1.5 * coef_norm, "k--", linewidth=2)

# coef_norm = Froude_numbers**-3
# ax1.loglog(Froude_numbers, Froude_numbers**3*coef_norm, 'k--', linewidth=2)

ax1.loglog(Froude_numbers, eps / eps_tot * coef_norm, "kx", linewidth=2)


plt.show()
