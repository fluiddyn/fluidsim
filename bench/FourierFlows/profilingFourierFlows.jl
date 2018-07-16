# profilingFourierFlows.jl

# This script integrates the 2D Navier-Stokes equation for nsteps time-steps
# with a 4th-order Runge Kutta time-stepping method and a pseudospectral
# method for performing spatial derivatives, using the FourierFlows.jl
# package.

using FourierFlows
using StatProfilerHTML
import FourierFlows.TwoDTurb


# Numerical parameters
n = 512*2              # 2D resolution = n^2
stepper = "RK4"        # timestepper
nsteps = 10            # number of timesteps
nthreads = 1           # number of FFTW threads (choose Sys.CPU_CORES for the maximum on system)
effort = FFTW.MEASURE  # FFTW planning effort

# Physical parameters
L = 2π      # domain size
dt = 5e-3   # timestep
nu = 1e-5   # (hyper)diffusion coefficient
nnu = 1     # (hyper)diffusion order (0 = linear drag, 1 = Laplacian viscosity)
mu = 0.0    # hypodiffusion coefficient
nmu = 2     # hypodiffusion order

# Initialize problem
g = TwoDGrid(n, L; nthreads=nthreads, effort=effort)
p = TwoDTurb.Params(nu, nnu, mu, nmu)
v = TwoDTurb.Vars(g)
eq = TwoDTurb.Equation(p, g)
ts = FourierFlows.autoconstructtimestepper(stepper, dt, eq.LC, g)
prob = FourierFlows.Problem(g, v, p, eq, ts)

# Set initial condition
TwoDTurb.set_q!(prob, rand(n, n))

# Compile step
stepforward!(prob)

println("Profiling for $nsteps steps with $nthreads threads and $n^2 resolution:")

Profile.clear()
@profile stepforward!(prob, nsteps)
Profile.print(format=:tree, mincount=20)
statprofilehtml()
