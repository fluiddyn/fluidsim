"""Tiny benchmark of dedalus on the case of biperiodic ns2d
===========================================================

To be compared with::

  fluidsim-bench 512 512 -s ns2d -it 10

  mpirun -np 2 fluidsim-bench 512 512 -s ns2d -it 10

"""

import numpy as np

from dedalus import public as de
import time

lx, ly = (1.0, 1.0)
nx, ny = (512, 512)

# Create bases and domain
x_basis = de.Fourier("x", nx, interval=(0, lx), dealias=3 / 2)
y_basis = de.Fourier("y", ny, interval=(0, ly), dealias=3 / 2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Faster algorithm with vorticity?
variables = ["p", "u", "v", "uy", "vy"]
problem = de.IVP(domain, variables=variables)

Reynolds = 1e4
problem.parameters["Re"] = Reynolds

problem.add_equation(
    "dt(u) + dx(p) - 1/Re*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy"
)
problem.add_equation(
    "dt(v) + dy(p) - 1/Re*(dx(dx(v)) + dy(vy)) = - u*dx(v) - v*vy"
)

problem.add_equation("dx(u) + vy = 0")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")

ts = de.timesteppers.RK443

solver = problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
u = solver.state["u"]
uy = solver.state["uy"]
v = solver.state["v"]
vy = solver.state["vy"]

u["g"] = np.ones_like(x)
v["g"] = np.ones_like(x)
u.differentiate("y", out=uy)
v.differentiate("y", out=vy)

dt = 1e-12

print("Starting loop")
start_time = time.time()

for it in range(10):
    solver.step(dt)

end_time = time.time()

print("Run time: %f" % (end_time - start_time))
