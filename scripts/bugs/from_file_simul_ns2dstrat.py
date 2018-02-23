"""
from_file_simul_ns2dstrat.py
============================

# Environment:
--------------
Python 3.6 (Conda)

# Description
--------------
Value of sum(transferEK_kx) too large in MPI calculations! 
The error is obtained for large resolutions nh = 1920 (MPI calculations
(server and pc).
The error is NOT obtained for resolutions nh = 480 and nh = 960 (MPI).
The error is obtained also for the repository fluiddyn/fluidsim.

# Result:
---------
warning: (abs(v.sum()) > small_value) for transferEK_kx
k =  transferEK_kx
abs(v.sum()) =  0.337343430975
warning: (abs(v.sum()) > small_value) for transferZ_2d
k =  transferZ_2d
abs(v.sum()) =  2.00277620009e-13
Computation completed in   13.051 s
path_run =
/fsnet/project/meige/2015/15DELDUCA/DataSim/tests/NS2D.strat_1920x1920_S8x8_F05_gamma05_2018-02-23_18-04-49
save state_phys in file state_phys_t003.006.nc

# Result expected
-----------------
transferEK_kx.sum() ~ 0 (or < 1e-14)

# Procedure to run the bug:
---------------------------
mpirun -np 4 python from_file_simul_ns2dstrat.py

"""

from __future__ import print_function

import os
from fluidsim.solvers.ns2d.strat.solver import Simul
from fluidsim.base.params import load_params_simul

path = '/fsnet/project/meige/2015/15DELDUCA/DataSim/tests/NS2D.strat_1920x1920_S8x8_F05_gamma05_2018-02-23_17-11-05'
file_name = 'state_phys_t003.002.nc'

path_file = os.path.join(path, file_name)

params = load_params_simul(path)
params.init_fields.type = 'from_file'
params.init_fields.from_file.path = path_file

params.time_stepping.t_end = 3.005
params.output.periods_save.spect_energy_budg = 1e-15

sim = Simul(params)
sim.time_stepping.start()
