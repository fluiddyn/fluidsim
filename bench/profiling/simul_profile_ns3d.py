#!/usr/bin/env python
"""
run simul_profile_ns3d.py
mpirun -np 8 python simul_profile_ns3d.py

"""

from fluidsim.solvers.ns3d import solver
from util_bench import profile, modif_params_profile3d

params = solver.Simul.create_default_params()
modif_params_profile3d(params, nh=128, nz=128)

sim = solver.Simul(params)

if __name__ == "__main__":
    profile(sim, nb_dim=3)


# params.oper.type_fft = 'fluidfft.fft3d.with_fftw3d'
# params.oper.type_fft = 'fluidfft.fft3d.with_cufft'
# params.oper.type_fft = 'fluidfft.fft3d.mpi_with_fftwmpi3d'
