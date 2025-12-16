"""
A small example of a script to save figures from a simulation directory
"""

import matplotlib.pyplot as plt

from pathlib import Path

from fluidsim import load

path_dir = (
    Path.home()
    / "Sim_data/tma/NS2D_forcing_const_energy_rate_128x128_S2pix2pi_2023-09-29_11-48-26"
)

sim = load(path_dir)

sim.output.spect_energy_budg.plot(tmin=4)

fig = plt.gcf()

fig.savefig(path_dir / "sim0_spectr_energy_budget.png")
