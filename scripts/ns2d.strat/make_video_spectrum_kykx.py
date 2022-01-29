"""
make_video_spectrum_kykx.py
===========================
Last modification : 17/01/2019

Makes animation energy spectra (kx, ky) evolution in time.

The two keys plotted are ap_fft and am_fft
"""
from pathlib import Path

import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt

from matplotlib import animation
from fluidsim import load_state_phys_file

# Parameters
key = "ap_fft"
kz_negative_enable = True
SAVE = True

# Create path with pathlib package.
root_dir = (
    Path("/fsnet/project/meige/2015/15DELDUCA/DataSim")
    / "isotropy_forcing_multidim"
)

list_simulations = [item for item in root_dir.glob("NS2D*")]

index = None
if key == "ap_fft":
    if kz_negative_enable:
        index = 0
    elif not kz_negative_enable:
        index = 3

elif key == "rot_fft":
    if kz_negative_enable:
        index = 1
    elif not kz_negative_enable:
        index = 2

if not index in np.arange(len(list_simulations)):
    raise ValueError("index should be defined.")

path = list_simulations[index]

### Parameters
skip = 2
tmin = 4
tmax = 400
scale = "log"  # can be "linear"

# Load object simulation
sim = load_state_phys_file(path.as_posix(), merge_missing_params=True)
poper = sim.params.oper
pforcing = sim.params.forcing

# Load data
with h5py.File((path / "spectra_multidim.h5").as_posix(), "r") as f:

    times = f["times"][...]
    itmin = np.argmin(abs(times - tmin))
    itmax = np.argmin(abs(times - tmax))
    times = times[itmin:itmax:skip]

    kx = f["kxE"][...]

    ap_fft_spectrum = f["spectrumkykx_ap_fft"]
    data_ap = ap_fft_spectrum[itmin:itmax:skip, :, :]

    am_fft_spectrum = f["spectrumkykx_am_fft"]
    data_am = am_fft_spectrum[itmin:itmax:skip, :, :]

# # Create array kz
kz = 2 * np.pi * np.fft.fftfreq(poper.ny, poper.Ly / poper.ny)
kz[kz.shape[0] // 2] *= -1
kz_modified = np.empty_like(kz)
kz_modified[0 : kz_modified.shape[0] // 2 - 1] = kz[
    kz_modified.shape[0] // 2 + 1 :
]
kz_modified[kz_modified.shape[0] // 2 - 1 :] = kz[
    0 : kz_modified.shape[0] // 2 + 1
]

# Create Grid
KX, KZ = np.meshgrid(kx, kz_modified)

# Modify data_ap
data_ap_plot = data_ap[0, :, :]
data_ap_plot_modified = np.empty_like(data_ap_plot)
data_ap_plot_modified[0 : kz_modified.shape[0] // 2 - 1, :] = data_ap_plot[
    kz_modified.shape[0] // 2 + 1 :, :
]
data_ap_plot_modified[kz_modified.shape[0] // 2 - 1 :, :] = data_ap_plot[
    0 : kz_modified.shape[0] // 2 + 1, :
]

# Modify data_am
data_am_plot = data_am[0, :, :]
data_am_plot_modified = np.empty_like(data_am_plot)
data_am_plot_modified[0 : kz_modified.shape[0] // 2 - 1, :] = data_am_plot[
    kz_modified.shape[0] // 2 + 1 :, :
]
data_am_plot_modified[kz_modified.shape[0] // 2 - 1 :, :] = data_am_plot[
    0 : kz_modified.shape[0] // 2 + 1, :
]

# Initialize figure
fig = plt.figure()
plt.rc("text", usetex=True)
ax = fig.add_subplot(221)
# ax.set_xlabel(r"$k_x$", fontsize=14)
ax.set_ylabel(r"$k_z$", fontsize=14)
ax.set_aspect("equal")

ax1 = fig.add_subplot(122)
ax1.set_xlabel(r"$t / \tau_{{af}}$", fontsize=14)
ax1.set_ylabel(r"$<E_a> / (P_a^2 l_f^{10})^{1/7}$", fontsize=14)

ax2 = fig.add_subplot(223)
ax2.set_xlabel(r"$k_x$", fontsize=14)
ax2.set_ylabel(r"$k_z$", fontsize=14)
ax2.set_aspect("equal")

# Cmpute mean dissipation
dict_spatial = sim.output.spatial_means.load()
times_spatial = dict_spatial["t"]
itmin_spatial = np.argmin(abs(times_spatial - 500))
eps = dict_spatial["epsK_tot"] + dict_spatial["epsA_tot"]
eps = eps[itmin_spatial:].mean(0)

# Compute energy forcing
forcing_rate = pforcing.forcing_rate
l_f = 2 * np.pi / (pforcing.nkmax_forcing * sim.oper.deltaky)
# energy_f = 100 * (forcing_rate ** 6 * l_f**2)**(1/7)
energy_f = ((forcing_rate**2) * (l_f**10)) ** (1 / 7)

# Compute energy
energies_ap = np.empty_like(times)
energies_am = np.empty_like(times)

for it, time in enumerate(times):
    energies_ap[it] = (
        (sim.oper.deltakx * sim.oper.deltaky) * data_ap[it].sum() / energy_f
    )
    energies_am[it] = (
        (sim.oper.deltakx * sim.oper.deltaky) * data_am[it].sum() / energy_f
    )

# Set axes limit
xlim = 200
ikx = np.argmin(abs(kx - xlim))
ax.set_xlim([0, kx[ikx] - sim.oper.deltakx])
ax2.set_xlim([0, kx[ikx] - sim.oper.deltakx])

zlim = 200
ikz = np.argmin(abs(kz - zlim))
ikz_negative = np.argmin(abs(kz + zlim))
ax.set_ylim([kz[ikz_negative], kz[ikz] - sim.oper.deltaky])
ax2.set_ylim([kz[ikz_negative], kz[ikz] - sim.oper.deltaky])

# Sec limits colormap

if scale == "linear":
    vmin = 0
    vmax = 0.001 * data_ap_plot_modified.max()

elif scale == "log":
    vmin = -5
    vmax = np.log10(data_ap_plot_modified.max()) - 3

ikx_text = np.argmin(abs(kx - kx[ikx] * 0.7))
ikz_text = np.argmin(abs(kz - kz[ikz] * 0.7))

# Plot first figure

ax.plot(kx, 1e-0 * sim.params.N * (kx / eps) ** (1 / 3), color="white")
ax.plot(kx, -1e-0 * sim.params.N * (kx / eps) ** (1 / 3), color="white")
ax.text(kx[ikx_text], kz[ikz_text], r"\hat{a}_+", color="white", fontsize=15)

ax2.plot(kx, 1e-0 * sim.params.N * (kx / eps) ** (1 / 3), color="white")
ax2.plot(kx, -1e-0 * sim.params.N * (kx / eps) ** (1 / 3), color="white")
ax2.text(kx[ikx_text], kz[ikz_text], r"\hat{a}_-", color="white", fontsize=15)

if scale == "linear":
    _im = ax.pcolormesh(KX, KZ, data_ap_plot_modified, vmin=vmin, vmax=vmax)
    _im2 = ax2.pcolormesh(KX, KZ, data_am_plot_modified, vmin=vmin, vmax=vmax)
elif scale == "log":
    _im = ax.pcolormesh(
        KX, KZ, np.log10(data_ap_plot_modified), vmin=vmin, vmax=vmax
    )
    _im2 = ax2.pcolormesh(
        KX, KZ, np.log10(data_am_plot_modified), vmin=vmin, vmax=vmax
    )

ax1.plot(times, energies_ap, color="grey", alpha=0.4)
ax1.plot(times, energies_am, color="grey", alpha=0.4)

_im_inset = ax1.plot(times[0], energies_ap[0], color="red", label=r"$\hat{a}_+$")
_im_inset3 = ax1.plot(
    times[0], energies_am[0], color="blue", label=r"$\hat{a}_-$"
)
ax1.legend(fontsize=14)
cbar_ax = fig.add_axes([0.38, 0.15, 0.01, 0.7])
colorbar = fig.colorbar(_im, cax=cbar_ax, format="%.1f")
# colorbar.ax.ticklabel_format(style="sci")

# ax1.plot(times[0:2], energies_ap[0:2], color="red")
# Define iterative function _update
def _update(frame):
    data_ap_plot = data_ap[frame, :, :]
    data_ap_plot_modified = np.empty_like(data_ap_plot)
    data_ap_plot_modified[0 : kz_modified.shape[0] // 2 - 1, :] = data_ap_plot[
        kz_modified.shape[0] // 2 + 1 :, :
    ]
    data_ap_plot_modified[kz_modified.shape[0] // 2 - 1 :, :] = data_ap_plot[
        0 : kz_modified.shape[0] // 2 + 1, :
    ]

    data_am_plot = data_am[frame, :, :]
    data_am_plot_modified = np.empty_like(data_am_plot)
    data_am_plot_modified[0 : kz_modified.shape[0] // 2 - 1, :] = data_am_plot[
        kz_modified.shape[0] // 2 + 1 :, :
    ]
    data_am_plot_modified[kz_modified.shape[0] // 2 - 1 :, :] = data_am_plot[
        0 : kz_modified.shape[0] // 2 + 1, :
    ]

    # Trick: https://stackoverflow.com/questions/29009743/using-set-array-with-pyplot-pcolormesh-ruins-figure
    if scale == "linear":
        _im.set_clim(vmin=0, vmax=0.001 * data_ap_plot_modified.max())
        _im.set_array(data_ap_plot_modified[:-1, :-1].flatten())

        _im2.set_clim(vmin=0, vmax=0.001 * data_ap_plot_modified.max())
        _im2.set_array(data_am_plot_modified[:-1, :-1].flatten())

    elif scale == "log":
        _im.set_clim(vmin=-5, vmax=np.log10(data_ap_plot_modified.max()) - 3)
        _im.set_array(np.log10(data_ap_plot_modified[:-1, :-1].flatten()))

        _im2.set_clim(vmin=-5, vmax=np.log10(data_ap_plot_modified.max()) - 3)
        _im2.set_array(np.log10(data_am_plot_modified[:-1, :-1].flatten()))

    _im_inset[0].set_data(times[0:frame], energies_ap[0:frame])
    _im_inset3[0].set_data(times[0:frame], energies_am[0:frame])

    ax.set_title(r"$t = {:.0f} \tau_{{af}}$".format(times[frame]), fontsize=14)


ani = animation.FuncAnimation(
    fig, _update, len(times), interval=1000, repeat=True
)

if SAVE:
    ani.save(
        "/home/users/calpelin7m/Phd/Movies_spectrakzkx/spectrumkykx_{}_kznegative_{}.mp4".format(
            sim.params.forcing.key_forced,
            bool(sim.params.forcing.tcrandom_anisotropic.kz_negative_enable),
        ),
        writer="ffmpeg",
    )
