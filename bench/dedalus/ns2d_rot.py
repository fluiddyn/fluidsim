"""Tiny benchmark of dedalus on the case of biperiodic ns2d
===========================================================

To run::

  python ns2d_rot.py


To be compared with::

  fluidsim-bench 512 -d 2 -s ns2d -it 10

  mpirun -np 2 fluidsim-bench 512 -d 2 -s ns2d -it 10

"""

import time
import numpy as np
from scipy.signal import gausspulse
from dedalus import public as de


lx, ly = (1.0, 1.0)
n = 1024
nx, ny = (n, n)

# Create bases and domain
x_basis = de.Fourier("x", nx, interval=(0, lx), dealias=3 / 2)
y_basis = de.Fourier("y", ny, interval=(0, ly), dealias=3 / 2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Stream function-vorticity formulation
variables = ["rot", "psi", "u", "v", "rotx", "roty"]
problem = de.IVP(domain, variables=variables)

Reynolds = 1e4
problem.parameters["Re"] = Reynolds

problem.add_equation("dt(rot) - (1/Re)*(dx(rotx) + dy(roty)) = - u*rotx - v*roty")
problem.add_equation("dx(dx(psi)) + dy(dy(psi)) + rot = 0")

# with first-order reduction equations...
# problem.add_equation('rot + dy(u) - dx(v) = 0')
problem.add_equation("rotx - dx(rot) = 0")
problem.add_equation("roty - dy(rot) = 0")
problem.add_equation("v + dx(psi) = 0")
problem.add_equation("u - dy(psi) = 0")

ts = de.timesteppers.RK443

solver = problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
rot = solver.state["rot"]
psi = solver.state["psi"]
u = solver.state["u"]
v = solver.state["v"]
rotx = solver.state["rotx"]
roty = solver.state["roty"]

# Initial conditions

# u['g'] = np.ones_like(x)
# v['g'] = np.ones_like(x)
# rot['g'] = -u.differentiate('y') + v.differentiate('x')
# rot.differentiate('x', out=rotx)
# rot.differentiate('y', out=roty)

rot["g"] = gausspulse(np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2), fc=1)
rot.differentiate("x", out=rotx)
rot.differentiate("y", out=roty)

# psi['g'] = 10 * y
# psi.differentiate('y', out=u)

dt = 1e-12

# Integration parameters
solver.stop_sim_time = np.inf
solver.stop_wall_time = np.inf
solver.stop_iteration = 10

print("Starting main time loop...")
start_time = time.time()

for it in range(10):
    solver.step(dt)

end_time = time.time()

print("Run time for the loop: %f" % (end_time - start_time))
