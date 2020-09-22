"""Wrong output it=0 with modifying the state after the init
============================================================

Bug description
---------------

- Values of outputs for t=0 are completely wrong.

Notes
-----

- Commit 797 introduces the the parameter `params.init_fields.modif_after_init`.

- pa: I think the bug is solved with commit 798.

To get the bugs
---------------

python wrong_output_it0_initfield.py

"""

import matplotlib.pyplot as plt

from fluidsim.solvers.ns2d.solver import Simul

params = Simul.create_default_params()

params.oper.nx = nx = 128

params.oper.ny = ny = nx // 2
params.oper.Lx = 10.0
params.oper.Ly = params.oper.Lx * (ny / nx)

params.nu_8 = 1e-10

params.init_fields.type = "noise"
params.init_fields.noise.velo_max = 1.0
params.init_fields.noise.length = 1.0
params.init_fields.modif_after_init = True

params.forcing.enable = True
params.forcing.type = "tcrandom"

params.forcing.nkmax_forcing = 10
params.forcing.nkmin_forcing = 4
params.forcing.tcrandom.time_correlation = 1e-10

params.time_stepping.USE_T_END = False
params.time_stepping.it_end = 5
# params.time_stepping.deltat_max = 1e-3

params.output.HAS_TO_SAVE = True
params.output.periods_save.spatial_means = 1e-10
params.output.periods_save.phys_fields = 1e-10

sim = Simul(params)

# we modify the state after the init...
sim.state.state_phys *= 2.0
sim.state.statespect_from_statephys()

sim.time_stepping.start()

sim.output.spatial_means.plot()
# sim.output.spatial_means.plot_dt_enstrophy()

plt.show()
