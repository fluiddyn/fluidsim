"""
movie.py
========
Make video physical fields of the simulation.
"""
from __future__ import print_function

from fluidsim import load_sim_for_plot
from glob import glob

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Make it easier for the user
path_sim = '/home/users/calpelin7m/Sim_data/' \
           'NS2D.strat_test_128x128_S2pix2pi_2017-10-02_15-01-54'

# Load simulation
sim = load_sim_for_plot(path_sim) 

# Parameters
nx = sim.params.oper.nx
nz = sim.params.oper.ny
Lx = sim.params.oper.Lx
Lz = sim.params.oper.Ly

# x, z axis
x = np.linspace(0, Lx, nx)
z = np.linspace(0, Lz, nz)

# GRID
X, Z = np.meshgrid(x, z)

# key to plot
key_plot = 'rot'

# Parameters of the plot
fig = plt.figure()
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title('{}, nh = {}'.format(key_plot, nx))

phys_state = glob(path_sim + '/state_*')


def animate(frame):
    """Function called by matplotlib to update each frame."""
    f = h5py.File(phys_state[frame], 'r')
    data = f['state_phys'][key_plot].value
    ux = f['state_phys']['ux'].value
    uz = f['state_phys']['uy'].value
    f.close()
    ax.pcolor(X, Z, data)
    ax.quiver(X, Z, ux, uz)
    
ani = animation.FuncAnimation(fig, animate, repeat=True,
                              frames=len(phys_state),
                              interval=200, blit=False)

