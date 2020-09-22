#!/usr/bin/env python
"""
python simul_profile2d.py
mpirun -np 2 python simul_profile2d.py

FLUIDSIM_NO_FLUIDFFT=1 python simul_profile2d.py

"""
from fluidsim import import_module_solver_from_key
from util_bench import profile, modif_params_profile2d

key = "ns2d"

solver = import_module_solver_from_key(key)

params = solver.Simul.create_default_params()
modif_params_profile2d(params)

sim = solver.Simul(params)

if __name__ == "__main__":
    profile(sim)
