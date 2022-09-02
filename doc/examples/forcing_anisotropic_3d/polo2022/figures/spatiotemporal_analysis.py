"""
    Plot four figures for each simulations and one key:
    1. spectra_vs_kzomega_slice_normalized
    2. spectra_vs_khomega_slice_normalized
    3. omega_emp
    4. nonlinear_broadening
"""
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
from pathlib import Path
import glob
import h5py
import os
from matplotlib.collections import LineCollection
import re

from fluiddyn.util import modification_date
from fluidsim.util import times_start_last_from_path, load_params_simul
from fluidsim import load

from util import (
    path_base_jeanzay,
    get_t_end,
    get_t_statio,
    couples,
)


nh = 1280
keys = ["potential", "kinetic", "total"]
key = "total"
cm = matplotlib.cm.get_cmap("jet", 100)
couples1280 = couples[1280]

path_output = "/linkhome/rech/genlag01/uey73qw/spatiotemporal_analysis/"  # Path where the figures are saved

print(path_base_jeanzay)
paths = sorted(path_base_jeanzay.glob(f"ns3d*_polo_*_{nh}x{nh}x*"))
print(paths)


def compute_omega_emp_vs_kzkh(
    spectrum,
    kh_spectra,
    kz_spectra,
    omegas,
):
    r"""Compute empirical frequency and fluctuation from the spatiotemporal spectra:

    .. math::

      \omega_{emp}(k_h, k_z) =
        \frac{\int ~ \omega ~ S(k_h, k_z, \omega)
        ~ \mathrm{d}\omega}{\int ~ S(k_h, k_z, \omega) ~ \mathrm{d}\omega},

      \delta \omega_{emp}(k_h, k_z) =
        \sqrt{\frac{\int ~ (\omega - \omega_{emp})^2 ~ S(k_h, k_z, \omega)
        ~ \mathrm{d}\omega}{\int ~ S(k_h, k_z, \omega) ~ \mathrm{d}\omega}},

    where :math:`\omega_{emp}` is the empirical frequency and :math:`\delta
    \omega_{emp}` is the empirical frequency fluctuation.
    """

    # khv, kzv = np.meshgrid(kh_spectra, kz_spectra)
    omega_emp = np.zeros((len(kz_spectra), len(kh_spectra)))
    delta_omega_emp = np.zeros((len(kz_spectra), len(kh_spectra)))
    omega_norm = np.zeros((len(kz_spectra), len(kh_spectra)))

    # we compute omega_emp first
    for io in range(len(omegas)):
        omega_emp += omegas[io] * spectrum[:, :, io]
        omega_norm += spectrum[:, :, io]
    omega_emp = omega_emp / omega_norm

    # then we conpute delta_omega_emp
    for io in range(len(omegas)):
        delta_omega_emp += ((omegas[io] - omega_emp) ** 2) * spectrum[:, :, io]
    delta_omega_emp = (np.divide(delta_omega_emp, omega_norm)) ** 0.5
    return omega_emp, delta_omega_emp


def spectra_vs_kzomega_slice_normalized(
    spectrum,
    kz_spectra,
    omegas,
    ikh,
):
    spectrum_kzomega = spectrum[:, ikh, :]
    spectrum_normalized = np.zeros(spectrum_kzomega.shape)

    for ikz in range(len(kz_spectra)):
        norm = sum(spectrum_kzomega[ikz, :])
        for io in range(len(omegas)):
            spectrum_normalized[ikz, io] = spectrum_kzomega[ikz, io] / norm
    return spectrum_normalized


def spectra_vs_khomega_slice_normalized(
    spectrum,
    kh_spectra,
    omegas,
    ikz,
):
    spectrum_khomega = spectrum[ikz, :, :]
    spectrum_normalized = np.zeros(spectrum_khomega.shape)

    for ikh in range(len(kh_spectra)):
        norm = sum(spectrum_khomega[ikh, :])
        for io in range(len(omegas)):
            spectrum_normalized[ikh, io] = spectrum_khomega[ikh, io] / norm
    return spectrum_normalized


for path in paths:
    t_start, t_last = times_start_last_from_path(path)

    params = load_params_simul(path)
    N = float(params.N)
    Rb = float(re.search(r"_Rb(.*?)_", path.name).group(1))
    nx = params.oper.nx
    proj = params.projection
    t_end = get_t_end(N, nh)
    t_statio = get_t_statio(N, nh)

    # Simulations with nu = 0 where just for testing on Licallo
    if params.nu_2 == 0.0:
        print(f"{path.name:90s} corresponds to a simulation with nul viscosity)")
        continue

    if t_last < t_end:
        print(f"{path.name:90s} not finished ({t_last=})")
        continue
    print(f"{path.name:90s} done ({t_last=})")

    path_spec = sorted(path.glob(f"spatiotemporal/periodogram_[0-9]*.h5"))

    if len(path_spec) != 1:
        print(f"Not only 1 periodogram in {path} \n")
        continue

    path_spec = path_spec[0]

    with h5py.File(path_spec, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        kh = f["kh_spectra"][:]
        kz = f["kz_spectra"][:]
        omegas = f["omegas"][:]
        print(omegas.shape, "\n \n")

        if key == "potential":
            spectrum = f["spectrum_A"][:]  # Potential energy
            key_tex = "E_A"
        elif key == "kinetic":
            spectrum = f["spectrum_K"][:]
            key_tex = "E_K"
        elif key == "total":
            spectrum = f["spectrum_A"][:] + f["spectrum_K"][:]
            key_tex = "E_A + E_K"
        else:
            print(f"Do not know key {key} \n")
            continue

        # print(spectrum.shape)

        #######################################
        # spectra_vs_kzomega_slice_normalized #
        #######################################
        ikh = 12
        spectra_normalized = spectra_vs_kzomega_slice_normalized(
            spectrum, kz, omegas, ikh
        )

        fig, ax = plt.subplots()
        ax.set_xlabel(r"$k_z/ \delta k_z$")
        ax.set_ylabel(r"$\omega / N$")
        xaxis = np.arange(kz.size)
        yaxis = omegas / N

        im = ax.pcolormesh(
            xaxis,
            yaxis,
            np.log10(spectra_normalized.transpose()),
            cmap=cm,
            vmin=-6,
            vmax=0,
            shading="nearest",
        )
        cbar = fig.colorbar(im)

        ax.plot(xaxis, kh[ikh] / ((kh[ikh] ** 2 + kz**2) ** 0.5), "k-")

        ax.set_title(
            rf"$proj={proj}  ~~~  N={N}  ~~~  {key_tex} ~~~ k_h / \delta k_h={ikh}$"
        )
        cbar.set_label(
            r"$\log \left( S(k_z, \omega) / \int S(k_z, \omega) \mathrm{d}\omega \right)$"
        )
        # cbar.set_label(r"$\log S(k_z, \omega)$")
        plt.savefig(
            f"{path_output}spectra_vs_kzomega_slice_normalized_ikh{ikh}_proj{proj}_N{N}_Rb{Rb}_{key}_nh{nh}.png",
            dpi=100,
        )

        #######################################
        # spectra_vs_khomega_slice_normalized #
        #######################################
        ikz = 12
        spectra_normalized = spectra_vs_khomega_slice_normalized(
            spectrum, kh, omegas, ikz
        )

        fig, ax = plt.subplots()
        ax.set_xlabel(r"$k_h/ \delta k_h$")
        ax.set_ylabel(r"$\omega / N$")
        xaxis = np.arange(kh.size)
        yaxis = omegas / N
        # plt.ylim([5e-8,1e-1])

        im = ax.pcolormesh(
            xaxis,
            yaxis,
            np.log10(spectra_normalized.transpose()),
            cmap=cm,
            vmin=-6,
            vmax=0,
            shading="nearest",
        )
        cbar = fig.colorbar(im)

        ax.plot(xaxis, kh / ((kh**2 + kz[ikz] ** 2) ** 0.5), "k-")

        ax.set_title(
            rf"$proj={proj}  ~~~  N={N}  ~~~  {key_tex} ~~~ k_z / \delta k_z={ikz}$"
        )
        cbar.set_label(
            r"$\log \left( S(k_h, \omega) / \int S(k_h, \omega) \mathrm{d}\omega \right)$"
        )
        # cbar.set_label(r"$\log S(k_h, \omega)$")
        plt.savefig(
            f"{path_output}spectra_vs_khomega_slice_normalized_ikz{ikz}_proj{proj}_N{N}_Rb{Rb}_{key}_nh{nh}.png",
            dpi=100,
        )

        #############
        # omega_emp #
        #############
        omega_emp, delta_omega_emp = compute_omega_emp_vs_kzkh(
            spectrum, kh, kz, omegas
        )
        KH, KZ = np.meshgrid(kh, kz)
        omega_disp = N * KH / ((KH**2 + KZ**2) ** 0.5)

        fig, ax = plt.subplots()
        ax.set_xlabel(r"$k_h/ \delta k_h$")
        ax.set_ylabel(r"$k_z/ \delta k_z$")
        xaxis = np.arange(kh.size)
        yaxis = np.arange(kz.size)

        im = ax.pcolormesh(
            xaxis,
            yaxis,
            ((omega_emp - omega_disp) / N),
            cmap=cm,
            vmin=-1,
            vmax=+1,
            shading="nearest",
        )
        cbar = fig.colorbar(im)

        ax.set_title(rf"$proj={proj}  ~~~  N={N}  ~~~  {key_tex}$")

        cbar.set_label(r"$(\omega_{emp}- \omega_L)/N$")
        plt.savefig(
            f"{path_output}omega_emp_proj{proj}_N{N}_Rb{Rb}_{key}_nh{nh}.png",
            dpi=100,
        )

        ########################
        # nonlinear_broadening #
        ########################
        omega_emp, delta_omega_emp = compute_omega_emp_vs_kzkh(
            spectrum, kh, kz, omegas
        )
        KH, KZ = np.meshgrid(kh, kz)
        omega_disp = N * KH / ((KH**2 + KZ**2) ** 0.5)

        fig, ax = plt.subplots()
        ax.set_xlabel(r"$k_h/ \delta k_h$")
        ax.set_ylabel(r"$k_z/ \delta k_z$")
        xaxis = np.arange(kh.size)
        yaxis = np.arange(kz.size)

        im = ax.pcolormesh(
            xaxis,
            yaxis,
            delta_omega_emp / omega_emp,
            cmap=cm,
            vmin=0,
            vmax=+2,
            shading="nearest",
        )
        cbar = fig.colorbar(im)

        ax.set_title(rf"$proj={proj}  ~~~  N={N}  ~~~  {key_tex}$")

        cbar.set_label(r"$\delta \omega_{emp}/ \omega_{emp}$")
        plt.savefig(
            f"{path_output}nonlinear_broadening_proj{proj}_N{N}_Rb{Rb}_{key}_nh{nh}.png",
            dpi=100,
        )
