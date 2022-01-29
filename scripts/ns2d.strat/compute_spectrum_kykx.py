"""
compute_spectrum_kykx.py
========================
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from math import pi, ceil
import matplotlib.patches as patches
from fluiddyn.output.rcparams import set_rcparams
from fluidsim import load_state_phys_file

# rot_fft tmin=200
# path = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-11-26_15-56-33"
# path = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing2/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-11-28_08-57-58"

# rot_fft kz_negative
# path = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-11-26_15-56-59"

## ap_fft tmin = 160
# path = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-11-26_15-57-14"

## ap_fft kz tmin= 200
# path = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing/NS2D.strat_1920x480_S2pix1.571_F07_gamma1_2018-11-26_15-57-25"

path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing/"
path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing2/"
path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim/isotropy_forcing3/"

index_simulation = 3
path = glob(os.path.join(path_root, "NS2D*"))[index_simulation]

# Create list path files
paths_files = glob(os.path.join(path, "state_phys*"))

# Create array of times
times = []
for path_file in paths_files:
    times.append(float(path_file.split("_t")[1].split(".nc")[0]))

times = np.asarray(times)
tmin = 2450
tmax = 2452
key = "ap"
SAVE = True
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
        ux = f["state_phys"]["ux"][...]
        uz = f["state_phys"]["uy"][...]
        b = f["state_phys"]["b"][...]

        # Fourier transform of the variables...
        ux_fft_arr[ifile, :, :] = np.fft.fft2(ux)
        uz_fft_arr[ifile, :, :] = np.fft.fft2(uz)
        b_fft_arr[ifile, :, :] = np.fft.fft2(b)
        ap_fft_arr[ifile, :, :] = N**2 * np.fft.fft2(
            uz
        ) + 1j * omega_k * np.fft.fft2(b)
        am_fft_arr[ifile, :, :] = N**2 * np.fft.fft2(
            uz
        ) - 1j * omega_k * np.fft.fft2(b)

# Time average
ux_fft_arr = np.mean(ux_fft_arr, axis=0)
uz_fft_arr = np.mean(uz_fft_arr, axis=0)
b_fft_arr = np.mean(b_fft_arr, axis=0)
ap_fft_arr = np.mean(ap_fft_arr, axis=0)
am_fft_arr = np.mean(am_fft_arr, axis=0)

# Parameters figure
set_rcparams(fontsize=14, for_article=True)
fig1, ax1 = plt.subplots()

ax1.set_xlabel("$k_x$", fontsize=20)
ax1.set_ylabel("$k_z$", fontsize=20)


ax1.tick_params(labelsize=16)
# ax1.set_title(r"$|\hat{a}_+|^2$")

# ax1.set_xlim([0,
#               sim.params.oper.coef_dealiasing * kx.max()//2])
# ax1.set_ylim([-sim.params.oper.coef_dealiasing * kz.max()//2,
#                sim.params.oper.coef_dealiasing * kz.max()//2])

# data = abs(ux_fft_arr)**2 + abs(uz_fft_arr)**2
# data = abs(b_fft_arr)**2
# data = abs(ap_fft_arr)**2
# data = abs(am_fft_arr)**2

if key == "ap":
    data = np.log10(abs(ap_fft_arr) ** 2)
    text = r"$\hat{a}_+$"
elif key == "am":
    data = np.log10(abs(am_fft_arr) ** 2)
    text = r"$\hat{a}_-$"
else:
    raise ValueError("Not implemented")


### Data
ikx = np.argmin(abs(kx - 200))
ikz = np.argmin(abs(kz - 148))
ikz_negative = np.argmin(abs(kz + 148))


ax1.set_xlim([0, kx[ikx] - sim.oper.deltakx])
ax1.set_ylim([kz[ikz_negative], kz[ikz] - sim.oper.deltaky])

# Plot kx > 0 and kz > 0
# ax1.pcolormesh(
#     KX[0:ikz, 0:ikx],
#     KZ[0:ikz, 0:ikx],
#     data[0:ikz, 0:ikx],
#     vmin=0,
#     vmax=1e7
# )

ax1.pcolormesh(
    KX[0:ikz, 0:ikx], KZ[0:ikz, 0:ikx], data[0:ikz, 0:ikx], vmin=5, vmax=8
)

# # Plot kx > 0 and kz < 0
KX_grid2 = KX[ikz_negative - 1 :, 0:ikx]

KZ_grid2 = np.empty_like(KX_grid2)
KZ_grid2[0:-1, :] = KZ[ikz_negative:, 0:ikx]
KZ_grid2[-1, :] = KZ[0, 0:ikx]

data_grid2 = np.empty_like(KX_grid2)
data_grid2[0:-1, :] = data[ikz_negative:, 0:ikx]
data_grid2[-1, :] = data[0, 0:ikx]

ax1.pcolormesh(KX_grid2, KZ_grid2, data_grid2, vmin=5, vmax=8)

##############

# ax1.pcolormesh(
#     KX[0:KZ.shape[0]//2, 0:KX.shape[1]//2],
#     KZ[0:KZ.shape[0]//2, 0:KX.shape[1]//2],
#     data[0:KZ.shape[0]//2, 0:KX.shape[1]//2],
#     vmin=0,
#     vmax=1e7
# )

# ax1.pcolormesh(
#     KX[KZ.shape[0]//2:, KX.shape[1]//2:],
#     KZ[KZ.shape[0]//2:, KX.shape[1]//2:],
#     data[KZ.shape[0]//2:, KX.shape[1]//2:],
#     vmin=0,
#     vmax=1e7
# )

# ax1.pcolormesh(
#     KX[KZ.shape[0]//2:, 0:KX.shape[1]//2],
#     KZ[KZ.shape[0]//2:, 0:KX.shape[1]//2],
#     data[KZ.shape[0]//2:, 0:KX.shape[1]//2],
#     vmin=0,
#     vmax=1e7
# )

# ax1.pcolormesh(
#     KX[0:KZ.shape[0]//2, KX.shape[1]//2:],
#     KZ[0:KZ.shape[0]//2, KX.shape[1]//2:],
#     data[0:KZ.shape[0]//2, KX.shape[1]//2:],
#     vmin=0,
#     vmax=1e7
# )

# Create a Rectangle patch
deltak = max(sim.oper.deltakx, sim.oper.deltaky)

x_rect = (
    np.sin(sim.params.forcing.tcrandom_anisotropic.angle)
    * deltak
    * sim.params.forcing.nkmin_forcing
)

z_rect = (
    np.cos(sim.params.forcing.tcrandom_anisotropic.angle)
    * deltak
    * sim.params.forcing.nkmin_forcing
)

width = abs(
    x_rect
    - np.sin(sim.params.forcing.tcrandom_anisotropic.angle)
    * deltak
    * sim.params.forcing.nkmax_forcing
)

height = abs(
    z_rect
    - np.cos(sim.params.forcing.tcrandom_anisotropic.angle)
    * deltak
    * sim.params.forcing.nkmax_forcing
)

rect1 = patches.Rectangle(
    (x_rect, z_rect), width, height, linewidth=1, edgecolor="r", facecolor="none"
)

ax1.add_patch(rect1)

if sim.params.forcing.tcrandom_anisotropic.kz_negative_enable:
    rect2 = patches.Rectangle(
        (x_rect, -(z_rect + height)),
        width,
        height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax1.add_patch(rect2)


# Plot arc kmin and kmax forcing
ax1.add_patch(
    patches.Arc(
        xy=(0, 0),
        width=2 * sim.params.forcing.nkmin_forcing * deltak,
        height=2 * sim.params.forcing.nkmin_forcing * deltak,
        angle=0,
        theta1=-90.0,
        theta2=90.0,
        linestyle="-.",
        color="red",
    )
)
ax1.add_patch(
    patches.Arc(
        xy=(0, 0),
        width=2 * sim.params.forcing.nkmax_forcing * deltak,
        height=2 * sim.params.forcing.nkmax_forcing * deltak,
        angle=0,
        theta1=-90,
        theta2=90.0,
        linestyle="-.",
        color="red",
    )
)

ax1.text(175, 120, text, fontsize=20, color="white")

ax1.set_aspect("equal")

if SAVE:
    path_save = "/home/users/calpelin7m/Phd/docs/Manuscript/figures"
    name = f"spectrakykx_forced{index_simulation}_key_{key}_t_{tmin}.png"
    fig1.savefig(os.path.join(path_save, name), bbox_inches="tight")
plt.show()
