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
variables = ["xi", "b", "u", "w", "uz", "wz", "xiz", "bz"]
problem = de.IVP(domain, variables=variables)

# Non-dimensional parameters
Reynolds = 1e4
Froude = 5e-1
Ratio_omegas = 1.0
Schmidt = 1.0

problem.parameters["Re"] = Reynolds
problem.parameters["F"] = Froude
problem.parameters["R"] = Ratio_omegas
problem.parameters["Sc"] = Schmidt

# Non-dimensional NS2D of stratified fluid.
problem.add_equation(
    "dt(xi) + (R**2 / (F * (1 - F**2)**(1/2.))) * dx(b) - (F**2  /  Re) * dx(dx(xi)) - ((1 - F**2) / Re) * dz(xiz) = - u * dx(xi) - w * xiz"
)

problem.add_equation(
    "dt(b) + uz - (F**2/(Re * Sc)) * dx(dx(b) - ((1 - F**2)/(Re * Sc))*dz(bz)) = - u*dx(b) - w*bz"
)

problem.add_equation("dx(u) + wz = 0")
problem.add_equation("xi - dz(u) + dx(w) = 0")

# with first-order reduction equations...
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("xiz - dz(xi) = 0")

ts = de.timesteppers.RK443

solver = problem.build_solver(ts)

x = domain.grid(0)
z = domain.grid(1)
xi = solver.state["xi"]
xiz = solver.state["xiz"]
u = solver.state["u"]
uz = solver.state["uz"]
w = solver.state["w"]
wz = solver.state["wz"]
b = solver.state["b"]
bz = solver.state["bz"]

u["g"] = np.ones_like(x)
w["g"] = np.ones_like(x)
b["g"] = np.ones_like(x)
xi["g"] = np.ones_like(x)
u.differentiate("z", out=uz)
w.differentiate("z", out=wz)
b.differentiate("z", out=bz)
xi.differentiate("z", out=xiz)

dt = 1e-12

print("Starting loop")
start_time = time.time()

for it in range(10):
    solver.step(dt)

end_time = time.time()

print("Run time: %f" % (end_time - start_time))
