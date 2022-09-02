#!/usr/bin/env python3
"""
Examples of commands:

```
python simul_ns3d_kolmo.py
python simul_ns3d_kolmo.py normalized
mpirun -np 4 python simul_ns3d_kolmo.py
```
"""

from math import pi
import sys

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.solver import Simul as SimulNotExtended

from fluidsim.base.forcing.kolmogorov import (
    extend_simul_class,
    KolmogorovFlow,
    KolmogorovFlowNormalized,
)
from fluidsim.base.output.horiz_means import HorizontalMeans


Simul = extend_simul_class(
    SimulNotExtended, [KolmogorovFlow, KolmogorovFlowNormalized, HorizontalMeans]
)

params = Simul.create_default_params()

params.output.sub_directory = "examples"
params.short_name_type_run = "kolmo"

params.oper.nx = params.oper.ny = nh = 64
params.oper.nz = nh

params.oper.Lx = params.oper.Ly = Lh = 10.0
params.oper.Lz = Lh * params.oper.nz / params.oper.nx

params.oper.coef_dealiasing = 2 / 3
params.time_stepping.t_end = 20.0

params.init_fields.type = "noise"
params.init_fields.noise.length = 1.0
params.init_fields.noise.velo_max = 1e-2

params.forcing.enable = True
params.forcing.type = "kolmogorov_flow"

if "normalized" in sys.argv:
    params.forcing.type += "_normalized"

params.forcing.kolmo.ik = 3
params.forcing.kolmo.amplitude = F = 1.0

L = params.oper.Lz / (2 * params.forcing.kolmo.ik)
injection_rate = (F * L) ** (3 / 2) / L

nh_target = nh
kmax = params.oper.coef_dealiasing * pi / Lh * nh_target
eta = 1 / kmax

order = 4
params.nu_4 = eta ** (order - 2 / 3) * injection_rate ** (1 / 3)
mpi.printby0(f"{params.nu_4 = :.2e}")

params.output.periods_print.print_stdout = 0.5
params.output.periods_save.phys_fields = 0.5
params.output.periods_save.spatial_means = 0.1
params.output.periods_save.spectra = 0.5
params.output.periods_save.spect_energy_budg = 0.5
params.output.periods_save.horiz_means = 0.1

sim = Simul(params)

sim.time_stepping.start()


mpi.printby0(
    "\nTo visualize the results, you can do:\n"
    f"cd {sim.output.path_run}; "
    + """ipython -i -c "from fluidsim import load; sim = load()"

sim.output.spatial_means.plot()

sim.output.phys_fields.set_equation_crosssection("y=0")
sim.output.phys_fields.animate('vx', dt_frame_in_sec=0.3, dt_equations=0.25)

tmin = 15
sim.output.horiz_means.plot(tmax=tmin)
sim.output.horiz_means.plot(tmin=tmin)

sim.output.spectra.plot1d(coef_compensate=5/3, tmin=tmin)
sim.output.spect_energy_budg.plot_fluxes(tmin=tmin)
"""
)
