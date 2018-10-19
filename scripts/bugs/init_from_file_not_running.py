"""
init_from_file_not_running.py
=============================

# Environment
--------------
Python 3.6.6 (Conda)

# Explanation
-------------
It does not compute when params.init_fields.type = "from_file" for
both sequential and MPI.

Bug in my terminal and cluster.

Both fluidfft and fluidsim are the last versions (tests OK).

# Traceback
------------
No traceback.

# To run the bug
----------------
sequential
python init_from_file_not_running.py

parallel (4 proc.)
mpirun -np 4 init_from_file_not_running.py

"""
from math import pi

import os
from glob import glob
from fluidsim.solvers.ns2d.strat.solver import Simul

# Create parameters
params = Simul.create_default_params()

# Operator parameters
params.oper.nx = nx = 960
params.oper.ny = nx // 4
params.oper.Lx = Lx = 2 * pi
params.oper.Ly = Lx / 4
params.oper.NO_SHEAR_MODES = True
params.oper.coef_dealiasing = 0.6666
params.oper.type_fft = None

# Without forcing
params.forcing.enable = False

# Parameters time stepping
params.time_stepping.USE_CFL = True
params.time_stepping.t_end = 2.
params.time_stepping.cfl_coef_group = None

# Parameters initialization
path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim"
paths_sims = sorted(glob(os.path.join(path_root, f"sim{nx}_no_shear_modes",
                                      "NS2D*")))
path_file = glob(paths_sims[0] + "/state_phys*")[-10]
params.init_fields.type = "from_file"
params.init_fields.from_file.path = path_file

# Parameters output
params.output.HAS_TO_SAVE = False
params.output.periods_print.print_stdout = 1e-2

# Launch...
sim = Simul(params)
sim.time_stepping.start()

