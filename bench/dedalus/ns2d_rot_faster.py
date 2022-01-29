"""Tiny benchmark of dedalus on the case of biperiodic ns2d
===========================================================

Optimized by Keaton Burns, see
https://bitbucket.org/dedalus-project/dedalus/issues/38/slow-simulation-ns2d-over-a-biperiodic

To run::

  python ns2d_rot_faster.py


To be compared with::

  fluidsim-bench 512 -d 2 -s ns2d -it 10

  mpirun -np 2 fluidsim-bench 512 -d 2 -s ns2d -it 10

"""

import time
import numpy as np
from scipy.signal import gausspulse
from dedalus import public as de


lx, ly = (1.0, 1.0)

nx = 512 * 2

coef_dealias = 2 / 3

n = int(coef_dealias * nx)
dealias = nx / n
nx, ny = (n, n)

# Create bases and domain
x_basis = de.Fourier("x", nx, interval=(0, lx), dealias=dealias)
y_basis = de.Fourier("y", ny, interval=(0, ly), dealias=dealias)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Stream function-vorticity formulation
variables = ["psi"]
problem = de.IVP(domain, variables=variables)

Reynolds = 1e4
problem.parameters["Re"] = Reynolds
problem.substitutions["u"] = "dy(psi)"
problem.substitutions["v"] = "-dx(psi)"
problem.substitutions["rot"] = "- dx(dx(psi)) - dy(dy(psi))"
problem.substitutions["rotx"] = "dx(rot)"
problem.substitutions["roty"] = "dy(rot)"
problem.add_equation(
    "dt(rot) - (1/Re)*(dx(rotx) + dy(roty)) = - u*rotx - v*roty",
    condition="(nx != 0) or (ny != 0)",
)
problem.add_equation("psi = 0", condition="(nx == 0) and (ny == 0)")

# with first-order reduction equations...
# problem.add_equation('rot + dy(u) - dx(v) = 0')

ts = de.timesteppers.RK443
# we could be faster "per time step" with SBDF3 but to be fair we need to use RK4
# ts = de.timesteppers.SBDF3

solver = problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
psi = solver.state["psi"]

# Initial conditions

# u['g'] = np.ones_like(x)
# v['g'] = np.ones_like(x)
# rot['g'] = -u.differentiate('y') + v.differentiate('x')
# rot.differentiate('x', out=rotx)
# rot.differentiate('y', out=roty)

rot = domain.new_field()
rot["g"] = gausspulse(np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2), fc=1)
kx = domain.elements(0)
ky = domain.elements(1)
k2 = kx**2 + ky**2
psi["c"][k2 != 0] = rot["c"][k2 != 0] / k2[k2 != 0]

# psi['g'] = 10 * y
# psi.differentiate('y', out=u)

dt = 1e-12

# Integration parameters
solver.stop_sim_time = np.inf
solver.stop_wall_time = np.inf
solver.stop_iteration = 10

print("Starting startup loop...")
start_time = time.time()
for it in range(10):
    solver.step(dt)
end_time = time.time()
print("Run time for startup loop: %f" % (end_time - start_time))

print("Starting main time loop...")
start_time = time.time()
for it in range(10):
    solver.step(dt)
end_time = time.time()
print("Run time for main loop: %f" % (end_time - start_time))
