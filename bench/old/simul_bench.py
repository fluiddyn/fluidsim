#!/usr/bin/env python
"""
python simul_bench.py
mpirun -np 2 python simul_bench.py

"""
from fluidsim import import_module_solver_from_key
from util_bench import modif_params_profile2d, modif_params_profile3d, bench

key = "ns2d"

solver = import_module_solver_from_key(key)

params = solver.Simul.create_default_params()
if "3d" in key:
    modif_params_profile3d(params)
else:
    modif_params_profile2d(params)

sim = solver.Simul(params)

if __name__ == "__main__":
    bench(sim)
