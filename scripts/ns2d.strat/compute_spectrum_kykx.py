"""
compute_spectrum_kykx.py
========================
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from math import pi

from fluidsim import load_state_phys_file

# rot_fft tmin=200
# path = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-11-26_15-56-33"

# rot_fft kz_negative
# path = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-11-26_15-56-59"

## ap_fft tmin = 160
# path = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-11-26_15-57-14"

## ap_fft kz tmin= 200
path = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-11-26_15-57-25"

# Create list path files
paths_files = glob(os.path.join(path, "state_phys*"))

# Create array of times
times = []
for path_file in paths_files:
    times.append(float(path_file.split("_t")[1].split(".nc")[0]))

times = np.asarray(times)
tmin = 100
tmax = 102
itmin = np.argmin(abs(times - tmin))
itmax = np.argmin(abs(times - tmax))


if not itmax:
    itmax = times.shape[0]

if itmax < itmin:
    raise ValueError("itmax should be larger than itmin")

# Load simulation
sim = load_state_phys_file(os.path.dirname(paths_files[-1]))

Lx = sim.params.oper.Lx
Lz = sim.params.oper.Ly
nx = sim.params.oper.nx
nz = sim.params.oper.ny
N = sim.params.N

# Array of wave-numbers in m^-1
kx = 2 * pi * np.fft.fftfreq(nx, Lx / nx)
kz = 2 * pi * np.fft.fftfreq(nz, Lz / nz)
KX, KZ = np.meshgrid(kx, kz)

omega_k = sim.params.N * (KX / np.sqrt(KX**2 + KZ**2))

# Create 3D_arrays
ux_fft_arr = np.empty([itmax - itmin, nz, nx], dtype="complex")
uz_fft_arr = np.empty([itmax - itmin, nz, nx], dtype="complex")
b_fft_arr = np.empty([itmax - itmin, nz, nx], dtype="complex")
ap_fft_arr = np.empty([itmax - itmin, nz, nx], dtype="complex")
am_fft_arr = np.empty([itmax - itmin, nz, nx], dtype="complex")

for ifile, path_file in enumerate(paths_files[itmin:itmax]):
    with h5py.File(path_file, "r") as f:
        ux = f["state_phys"]["ux"].value
        uz = f["state_phys"]["uy"].value
        b = f["state_phys"]["b"].value

        # Fourier transform of the variables...
        ux_fft_arr[ifile, :, :] = np.fft.fft2(ux)
        uz_fft_arr[ifile, :, :] = np.fft.fft2(uz)
        b_fft_arr[ifile, :, :] = np.fft.fft2(b)
        ap_fft_arr[ifile, :, :] = N**2 * np.fft.fft2(uz) - 1j * omega_k * np.fft.fft2(b)
        am_fft_arr[ifile, :, :] = N**2 * np.fft.fft2(uz) + 1j * omega_k * np.fft.fft2(b)

# Time average
ux_fft_arr = np.mean(ux_fft_arr, axis=0)
uz_fft_arr = np.mean(uz_fft_arr, axis=0)
b_fft_arr = np.mean(b_fft_arr, axis=0)
ap_fft_arr = np.mean(ap_fft_arr, axis=0)
am_fft_arr = np.mean(am_fft_arr, axis=0)

# Parameters figure
fig1, ax1 = plt.subplots()
ax1.set_xlabel("$k_x$")
ax1.set_ylabel("$k_z$")
ax1.set_title("abs(uz_fft_arr)**2 + abs(ux_fft_arr)**2")
ax1.set_xlim([-sim.params.oper.coef_dealiasing * kx.max(),
             sim.params.oper.coef_dealiasing * kx.max()])
ax1.set_ylim([-sim.params.oper.coef_dealiasing * kz.max(),
             sim.params.oper.coef_dealiasing * kz.max()])

data = abs(ux_fft_arr)**2 + abs(uz_fft_arr)**2
data = abs(b_fft_arr)**2
data = abs(ap_fft_arr)**2
data = abs(am_fft_arr)**2

ax1.pcolormesh(
    KX[0:KZ.shape[0]//2, 0:KX.shape[1]//2],
    KZ[0:KZ.shape[0]//2, 0:KX.shape[1]//2],
    data[0:KZ.shape[0]//2, 0:KX.shape[1]//2],
    vmin=0,
    vmax=1e6
)

ax1.pcolormesh(
    KX[KZ.shape[0]//2:, KX.shape[1]//2:],
    KZ[KZ.shape[0]//2:, KX.shape[1]//2:],
    data[KZ.shape[0]//2:, KX.shape[1]//2:],
    vmin=0,
    vmax=1e6
)

ax1.pcolormesh(
    KX[KZ.shape[0]//2:, 0:KX.shape[1]//2],
    KZ[KZ.shape[0]//2:, 0:KX.shape[1]//2],
    data[KZ.shape[0]//2:, 0:KX.shape[1]//2],
    vmin=0,
    vmax=1e6
)

ax1.pcolormesh(
    KX[0:KZ.shape[0]//2, KX.shape[1]//2:],
    KZ[0:KZ.shape[0]//2, KX.shape[1]//2:],
    data[0:KZ.shape[0]//2, KX.shape[1]//2:],
    vmin=0,
    vmax=1e6
)


# ax1.imshow(data, vmin=0, vmax=1e6)
# ax1.contourf(data)
# ax1.pcolormesh(
#     data,
#     vmin=0,
#     vmax=1e6
# )

# # Parameters figure
# fig2, ax2 = plt.subplots()
# ax2.set_xlabel("$k_x$")
# ax2.set_ylabel("$k_z$")
# ax2.set_title("abs(b_fft_arr)**2")
# ax2.set_xlim([-sim.params.oper.coef_dealiasing * kx.max(),
#              sim.params.oper.coef_dealiasing * kx.max()])
# ax2.set_ylim([-sim.params.oper.coef_dealiasing * kz.max(),
#              sim.params.oper.coef_dealiasing * kz.max()])

# data = abs(b_fft_arr)**2
# ax2.pcolormesh(
#     KX, KZ,
#     data,
#     vmin=0,
#     vmax=1e6
# )


# # Parameters figure
# fig3, ax3 = plt.subplots()
# ax3.set_xlabel("$k_x$")
# ax3.set_ylabel("$k_z$")
# ax3.set_title("abs(ap_fft_arr)**2")
# ax3.set_xlim([-sim.params.oper.coef_dealiasing * kx.max(),
#              sim.params.oper.coef_dealiasing * kx.max()])
# ax3.set_ylim([-sim.params.oper.coef_dealiasing * kz.max(),
#              sim.params.oper.coef_dealiasing * kz.max()])

# data = abs(ap_fft_arr)**2
# ax3.pcolormesh(
#     KX, KZ,
#     data,
#     vmin=0,
#     vmax=1e6
# )


plt.show()
