#!/usr/bin/env python
# coding=utf8

import numpy as np
import glob
import os

import matplotlib.pylab as plt

from solveq2d import solveq2d


# dir_base = 'Approach_runs_2048x2048'
# dir_base = 'Pure_standing_waves_512x512'

dir_base = (
    # '/scratch/augier/'
    "/home/pierre/"
    "Results_for_article_SW1l/"
    "Pure_standing_waves_1024x1024"
)


set_of_dir_results = solveq2d.SetOfDirResults(dir_base=dir_base)

values_c = np.array([10, 1000])

# values_c2 = set_of_dir_results.values_c2
values_c2 = values_c**2
values_solver = set_of_dir_results.values_solver
# values_solver = ['SW1l']

tmin = 80


tuple_loop = [
    (c2, name_solver) for c2 in values_c2 for name_solver in values_solver
]
for c2, name_solver in tuple_loop:
    path_dir = set_of_dir_results.one_path_from_values(
        solver=name_solver, c2=c2, FORCING=True
    )

    if path_dir is None:
        continue

    sim = solveq2d.load_state_phys_file(t_approx=2620, name_dir=path_dir)

    sim.output.spect_energy_budg.plot(tmin=tmin, tmax=500, delta_t=0.0)
    sim.output.spectra.plot2D(
        tmin=tmin, tmax=500, delta_t=0.0, coef_compensate=5.0 / 3
    )

    # sim.output.spatial_means.plot()
    sim.output.prob_dens_func.plot(tmin=tmin)

    sim.output.increments.plot(
        tmin=tmin, tmax=None, delta_t=0.0, order=6, yscale="log"
    )

    # sim.output.increments.plot_pdf(tmin=tmin, tmax=160.25,key_var='ux',
    #                                order=4)

    sim.output.increments.plot_Kolmo(tmin=tmin)

    # sim.output.phys_fields.plot(key_field='rot')
    # sim.output.phys_fields.plot(key_field='eta')


solveq2d.show()
