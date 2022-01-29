"""Simple spherical harmonic shallow water model toy code based on shtns
library.

Refs:

- "non-linear barotropically unstable shallow water test case"
 example provided by Jeffrey Whitaker
 https://gist.github.com/jswhit/3845307

- Galewsky et al (2004, Tellus, 56A, 429-440)
 "An initial-value problem for testing numerical models of the global
 shallow-water equations" DOI: 10.1111/j.1600-0870.2004.00071.x
 http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
 
- shtns/examples/shallow_water.py

- Jakob-Chien et al. 1995:
 "Spectral Transform Solutions to the Shallow Water Test Set"

"""
import numpy as np
from scipy import integrate
from fluidsim.solvers.sphere.sw1l.solver import Simul


params = Simul.create_default_params()

# Grid
N = 256
params.oper.lmax = N // 3
params.oper.nlat = N // 2
params.oper.nlon = N

hour = 60**2
params.time_stepping.USE_CFL = False
params.time_stepping.deltat0 = dt = 10  # 150 seconds for 3rd order AB scheme
params.time_stepping.t_end = 150 * hour

# Earth parameters
params.oper.radius = 6.37122e6  # meters
params.oper.omega = 7.292e-5  # rad/s
g = 9.80616  # acc. due to gravity m/s
hbar = 10.0e3  # depth of troposphere
params.c2 = g * hbar  # wave speed squared

# Viscosity
efold = 3.0 * hour  # efolding timescale at ntrunc for hyperdiffusion
params.nu_8 = params.oper.radius / efold

# I/O
params.init_fields.type = "in_script"
params.output.sub_directory = "examples"
params.output.periods_print.print_stdout = hour
# params.output.periods_save.phys_fields = hour
# TODO: save energy, enstrophy, q, u_rms, rot_rms?
# params.output.periods_save.spatial_means = 0.1
params.output.ONLINE_PLOT_OK = True
params.output.periods_plot.phys_fields = hour
params.output.phys_fields.field_to_plot = "rot"

sim = Simul(params)

# Initial condition: Zonal jet
umax = 80.0  # jet speed amplitude, m/s
oper = sim.oper

# Lat-Lon in radians
lons1r = oper.lons
lats1r = oper.lats
LONS = oper.LONS - np.pi
LATS = oper.LATS

# Initial fields: a zonal jet
phi0 = np.pi / 7.0
phi1 = 0.5 * np.pi - phi0
phi2 = 0.25 * np.pi


def ux_from_lats(phi):
    en = np.exp(-4.0 / (phi1 - phi0) ** 2)
    u1 = (umax / en) * np.exp(1.0 / ((phi - phi0) * (phi - phi1)))
    u0 = np.zeros_like(phi, np.float)
    return np.where(np.logical_and(phi < phi1, phi > phi0), u1, u0)


ug = ux_from_lats(lats1r)
ug.shape = (oper.nlat, 1)
ux = ug * oper.create_array_spat(value=1.0)  # broadcast to shape (nlats,nlons)

# uy = self.state_phys.get_var("uy")
def integrand_gh(phi, a, omega):
    """The balance equation (3) in Galewsky et al."""
    ux_phi = ux_from_lats(phi)
    f = 2 * omega * np.sin(phi)
    return a * ux_phi * (f + np.tan(phi) / a * ux_phi)


def integrate_gh(lower, upper):
    """Integrate the ``integrand_gh`` function from lower to upper limit."""
    return integrate.quad(
        integrand_gh,
        lower,
        upper,
        (params.oper.radius, params.oper.omega),
        #  maxiter=100
    )


def eta_from_h(h):
    # return g * (h - hbar) / params.c2
    return g * h / params.c2 - 1


def h_from_eta(eta):
    return params.c2 * (1 + eta) / g


phi_lower = lats1r.min()
gh1 = np.array([integrate_gh(phi_lower, float(phi)) for phi in lats1r])

gh1_error = gh1[:, 1]
h = hbar - gh1[:, 0] / g

etag = eta_from_h(h)

etag.shape = (oper.nlat, 1)
eta = etag * oper.create_array_spat(value=1.0)  # broadcast to shape (nlats,nlons)

# Height perturbation.
alpha = 1.0 / 3.0
beta = 1.0 / 15.0
hamp = 120.0  # amplitude of height perturbation to zonal jet
hbump = (
    hamp
    * np.cos(LATS)
    * np.exp(-((LONS / alpha) ** 2))
    * np.exp(-(((phi2 - LATS) / beta) ** 2))
)
eta += g * hbump / params.c2

sim.state.init_from_uxuyeta(ux, 0, eta)

# In this case (params.init_fields.type = 'in_script') if we want to plot the
# result of the initialization before the time_stepping, we need to manually
# initialized the output:
#
sim.output.init_with_initialized_state()
# TODO: parameter to disable / modify quiver plot
# sim.output.phys_fields._init_skip_quiver()
# sim.output.phys_fields._skip_quiver = oper.nlat


def visualize_fields2d():
    sim.output.phys_fields.plot("rot")
    sim.output.phys_fields.plot("div")
    sim.output.phys_fields.plot("eta")


def visualize_fields1d():
    """Verify if the initialization corresponds to Fig. 1 in Galewsky et al."""
    import matplotlib.pyplot as plt

    lats1d = np.degrees(lats1r)

    fig, axes = plt.subplots(1, 3, figsize=(9, 5), sharey=True)
    ax0, ax1, ax2 = axes.ravel()
    ax0.set_ylim(20, 70)
    ax0.set_ylabel("latitude (degrees)")

    # First subfigure
    ax0.plot(ug, lats1d)
    ax0.set_xlabel("Zonal wind (m/s)")
    ax0.set_xlim(None, 90)

    # Second subfigure
    ax1.plot(h, lats1d)
    ax1.set_xlabel("Height field (m)")

    # Third subfigure
    levels = np.arange(10, 100, 10)
    ax2.contour(np.degrees(LONS), np.degrees(LATS), hbump, levels=levels)
    ax2.set_xlim(-50, 50)
    ax2.set_title("h'")
    ax2.set_xlabel("longitude (degrees)")


if __name__ == "__main__":
    # visualize_fields1d()
    # visualize_fields2d()
    sim.time_stepping.start()
