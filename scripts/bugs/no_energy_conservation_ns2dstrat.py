"""
no_energy_conservation_ns2d_strat.py
=======================================

# Bug description
-----------------
No energy conservation (no viscosity and no forcing)
params.forcing.enable = False
params.nu_8 = 0

Injection of energy fluctuates.
params.forcing.enable = True

# Notes
--------
Same problem with fluiddyn/fluidsim

# To run the bug
--------------
python no_energy_conservation_ns2dstrat.py

"""
from __future__ import print_function

import numpy as np
from math import pi
import matplotlib.pyplot as plt

from fluidsim.solvers.ns2d.strat.solver import Simul

params = Simul.create_default_params()

params.oper.nx = nx = 64

params.oper.ny = ny = nx
params.oper.Lx = 2 * pi
params.oper.Ly = params.oper.Lx * (ny / nx)
params.oper.coef_dealiasing = 0.66

params.init_fields.type = 'noise'

params.forcing.enable = True
params.forcing.type = 'tcrandom_anisotropic'
# params.forcing.type = 'tcrandom'

params.forcing.nkmax_forcing = 8
params.forcing.nkmin_forcing = 4
params.forcing.tcrandom.time_correlation = 0.5
# params.forcing.random.only_positive = False
# params.forcing.normalized.which_root = 'minabs'
params.time_stepping.t_end = 2
# params.time_stepping.USE_CFL = False
# params.time_stepping.deltat0 = 0.005

params.output.HAS_TO_SAVE = True
params.output.periods_save.spatial_means = 1e-10
params.output.periods_save.phys_fields = 0.2
sim = Simul(params)
sim.time_stepping.start()

sim.output.spatial_means.plot()

d = sim.output.spatial_means.load()

Z = d['Z']
t = d['t']
Pz = d['PZ_tot']
dt_z = np.diff(Z)/np.diff(t)
Pz2 = d['PZ2']
Pz1 = d['PZ1']

E = d['E']
PK = d['PK_tot']
PK2 = d['PK2']
PK1 = d['PK1']
dt_E = np.diff(E)/np.diff(t)

fig, ax = plt.subplots()
ax.plot(t[:-1], dt_z)
ax.plot(t, Pz)
ax.plot(t, Pz1, 'r')
ax.plot(t, Pz2, 'g')
fig.suptitle('Z')


fig, ax = plt.subplots()
ax.plot(t[:-1], dt_E, 'k')
ax.plot(t, PK, '')
ax.plot(t, PK1, 'r')
ax.plot(t, PK2, 'g')
fig.suptitle('E')

plt.show()
