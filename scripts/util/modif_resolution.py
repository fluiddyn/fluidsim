#!/usr/bin/env python
# coding=utf8
#
# run modif_resolution.py
# python modif_resolution.py

from solveq2d import solveq2d

name_dir = (
    "~/Storage/Results_SW1lw/Pure_standing_waves_3840x3840/"
    + "SE2D_SW1lwaves_forcingw_L=50.x50._3840x3840_c=40_f=0_2013-07-23_15-47-25"
)
t_approx = 10.0e10
coef_modif_resol = 2

solveq2d.modif_resolution_from_dir(
    name_dir, t_approx=t_approx, coef_modif_resol=coef_modif_resol, PLOT=False
)
