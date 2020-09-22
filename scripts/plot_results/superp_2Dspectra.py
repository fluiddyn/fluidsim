#!/usr/bin/env python
# coding=utf8

import numpy as np
import glob
import os

import matplotlib.pylab as plt

from solveq2d import solveq2d


dir_base = "Approach_runs_2048x2048"
set_of_dir_results = solveq2d.SetOfDirResults(dir_base=dir_base)

values_c2 = set_of_dir_results.values_c2
values_c2 = [200]
values_solver = set_of_dir_results.values_solver
# values_solver = ['SW1l']

tmin = 263


tuple_loop = [
    (c2, name_solver) for c2 in values_c2 for name_solver in values_solver
]
for c2, name_solver in tuple_loop:
    path_dir = set_of_dir_results.one_path_from_values(solver=name_solver, c2=c2)

    sim = solveq2d.load_state_phys_file(t_approx=1000, name_dir=path_dir)

    sim.output.spect_energy_budg.plot(tmin=tmin, tmax=500, delta_t=0.0)
    sim.output.spectra.plot2D(
        tmin=tmin, tmax=500, delta_t=0.0, coef_compensate=5.0 / 3
    )

    sim.output.spatial_means.plot()
    # sim.output.prob_dens_func.plot(tmin=tmin)

    # sim.output.phys_fields.plot(key_field='rot')
    # sim.output.phys_fields.plot(key_field='eta')


plt.show()
