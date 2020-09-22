"""Tiny benchmark of dedalus on the case of biperiodic ns2d.strat
=================================================================

To be compared with::

  fluidsim-bench 512 512 -s ns2d -it 10

  mpirun -np 2 fluidsim-bench 512 512 -s ns2d -it 10

???? (## TODO: Bench for ns2d.strat)
"""

import numpy as np

from dedalus import public as de
import time

lx, lz = (1.0, 1.0)
nz, nx = (512, 512)

# Create bases and domain
x_basis = de.Fourier("x", nx, interval=(0, lx), dealias=3 / 2)
z_basis = de.Fourier("z", nz, interval=(0, lz), dealias=3 / 2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Faster algorithm with vorticity?
variables = ["p", "b", "u", "w", "uz", "wz", "bz"]
problem = de.IVP(domain, variables=variables)

# Non-dimensional parameters
Reynolds = 1e4
Froude_horiz = 5e-1
Schmidt = 1.0
Aspect_ratio = 1.0

problem.parameters["Re"] = Reynolds
problem.parameters["Fh"] = Froude_horiz
problem.parameters["Sc"] = Schmidt
problem.parameters["alpha"] = Aspect_ratio

# Non-dimensional NS2D of stratified fluid.
problem.add_equation(
    "dt(u) + dx(p) - (1/(Re * alpha**2)) * ((alpha**2) * dx(dx(u)) + dz(uz)) = - u*dx(u) - w*uz"
)
problem.add_equation(
    "(Fh**2) * dt(w) + dz(p) + b - (1/(Re * alpha**2)) * ((alpha**2) * dx(dx(w)) + dz(wz)) = (Fh**2) * (- u*dx(w) - w*wz)"
)

problem.add_equation(
    "dt(b) - uz - (1/(Re * Sc * alpha**2)) * ((alpha**2) * dx(dx(b)) + dz(bz)) = - u*dx(b) - w*bz"
)

problem.add_equation("dx(u) + (Fh**2 / alpha**2) * wz = 0")

# with first-order reduction equations...
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_equation("bz - dz(b) = 0")

ts = de.timesteppers.RK443

solver = problem.build_solver(ts)

x = domain.grid(0)
z = domain.grid(1)
u = solver.state["u"]
uz = solver.state["uz"]
w = solver.state["w"]
wz = solver.state["wz"]
b = solver.state["b"]
bz = solver.state["bz"]

u["g"] = np.ones_like(x)
w["g"] = np.ones_like(x)
b["g"] = np.ones_like(x)
u.differentiate("z", out=uz)
w.differentiate("z", out=wz)
b.differentiate("z", out=bz)

dt = 1e-12

print("Starting loop")
start_time = time.time()

for it in range(10):
    solver.step(dt)

end_time = time.time()

print("Run time: %f" % (end_time - start_time))
