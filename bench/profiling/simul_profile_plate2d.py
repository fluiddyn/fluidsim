#!/usr/bin/env python
"""
python simul_profile_plate2d.py
mpirun -np 2 python simul_profile_plate2d.py

FLUIDSIM_NO_FLUIDFFT=1 python simul_profile_plate2d.py
FLUIDSIM_NO_FLUIDFFT=1 mpirun -np 2 python simul_profile_plate2d.py

"""

from fluidsim.solvers.plate2d import solver
from util_bench import profile, modif_params_profile2d

params = solver.Simul.create_default_params()
modif_params_profile2d(params)


sim = solver.Simul(params)

if __name__ == "__main__":
    profile(sim)
