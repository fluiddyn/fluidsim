from pathlib import Path
import os

import numpy as np
from pandas import DataFrame

from fluiddyn.io.redirect_stdout import stdout_redirected

from fluidsim import load_sim_for_plot


def compute_ratio_diss2_disstot(sim, t_statio, hide_stdout=True):
    with stdout_redirected(hide_stdout):
        data = sim.output.spectra.load3d_mean(t_statio)
    spectrum = data["spectra_E"]
    k = data["k"]
    freq_nu_2 = sim.params.nu_2 * k**2
    freq_nu_4 = sim.params.nu_4 * k**4
    freq_diss = freq_nu_2 + freq_nu_4
    return (freq_nu_2 * spectrum).sum() / (freq_diss * spectrum).sum()


def compute_isotropy_velocities(sim, t_statio, hide_stdout=True):
    with stdout_redirected(hide_stdout):
        d = sim.output.spectra.load1d_mean(t_statio)

    kz = d["kz"]
    delta_kz = kz[1]
    EKx_kz = d["spectra_vx_kz"] * delta_kz
    EKy_kz = d["spectra_vy_kz"] * delta_kz
    EKz_kz = d["spectra_vz_kz"] * delta_kz

    EKx_kz[0] = 0
    EKy_kz[0] = 0

    EKz = EKz_kz.sum()
    EK = EKx_kz.sum() + EKy_kz.sum() + EKz

    return 3 * EKz / EK


def compute_isotropy_dissipation(sim, t_statio, hide_stdout=True):
    with stdout_redirected(hide_stdout):
        d = sim.output.spect_energy_budg.load_mean(t_statio)

    kh = d["kh"]
    kz = d["kz"]

    KH, KZ = np.meshgrid(kh, kz)
    assert np.allclose(KH[0, :], kh)
    assert np.allclose(KZ[:, 0], kz)
    K2 = KH**2 + KZ**2
    K4 = K2**2

    nu_2 = sim.params.nu_2
    nu_4 = sim.params.nu_4
    freq_diss = nu_2 * K2 + nu_4 * K4
    freq_diss[0, 0] = 1e-16

    diss_K = d["diss_Kh"] + d["diss_Kz"]
    assert diss_K.shape == KH.shape

    EK = diss_K / freq_diss

    epsK_kz = ((nu_2 * KZ**2 + nu_4 * KZ**4) * EK).sum()

    ratio = epsK_kz / diss_K.sum()

    ratio_iso = 0.215
    return (1 - ratio) / (1 - ratio_iso)


def get_time_last_saved_spatial_means(sim):

    path_run = Path(sim.output.path_run)
    with open(path_run / "spatial_means.txt", "rb") as file:
        file.seek(-1000, os.SEEK_END)
        txt = file.read().decode()

    return float(txt.split("####\ntime = ")[-1].split()[0])


def get_isotropy_values(sim, t_statio, recompute=False):

    first_line = (
        "# isotropy_velocity isotropy_dissipation ratio_diss2_disstot t_last"
    )

    t_last = get_time_last_saved_spatial_means(sim)

    path_run = Path(sim.output.path_run)
    path_file = path_run / "isotropy_values.txt"

    if path_file.exists() and not recompute:
        with open(path_file) as file:
            first_line_file = file.readline().strip()
            words = file.readline().split()

        if first_line != first_line_file or float(words[-1]) != t_last:
            return get_isotropy_values(sim, t_statio, recompute=True)

        isotropy_velocity = float(words[0])
        isotropy_dissipation = float(words[1])
        ratio_diss2_disstot = float(words[2])

        return isotropy_velocity, isotropy_dissipation, ratio_diss2_disstot

    print(f"compute for simulation\n{sim.output.path_run}")
    isotropy_velocity = compute_isotropy_velocities(sim, t_statio)
    isotropy_dissipation = compute_isotropy_dissipation(sim, t_statio)
    ratio_diss2_disstot = compute_ratio_diss2_disstot(sim, t_statio)

    with open(path_file, "w") as file:
        file.write(
            f"{first_line}\n{isotropy_velocity} "
            f"{isotropy_dissipation} {ratio_diss2_disstot} {t_last}\n"
        )

    return isotropy_velocity, isotropy_dissipation, ratio_diss2_disstot


def get_values(sim):
    N = sim.params.N
    # nu_4 = sim.params.nu_4

    from fluidsim.util import times_start_last_from_path

    t_start, t_last = times_start_last_from_path(sim.output.path_run)
    if t_last < 6:
        raise ValueError

    t_statio = max(6, t_start + 1)
    averages = sim.output.spatial_means.get_dimless_numbers_averaged(
        tmin=t_statio
    )

    U2 = averages["dimensional"]["Uh2"]
    epsK = averages["dimensional"]["epsK"]
    Gamma = averages["Gamma"]
    Fh = averages["Fh"]
    R2 = averages["R2"]
    try:
        R4 = averages["R4"]
    except KeyError:
        R4 = np.inf

    try:
        I_velocities, I_dissipation, ratio_diss2_disstot = get_isotropy_values(
            sim, t_statio
        )
    except OSError:
        print(sim.output.path_run)
        raise

    # fmt: off
    return [
        N, U2, epsK, Gamma, Fh, R2, R4,
        I_velocities, I_dissipation, ratio_diss2_disstot,
    ]
    # fmt: on


# fmt: off
columns = [
    "N",
    #"Rb", "Re",
    "U2", "epsK", "Gamma", "Fh", "R2", "R4",
    "I_velocities", "I_dissipation", "ratio_diss2_disstot",
]
# fmt: on


def get_dataframe_from_paths(paths):

    values = []
    for path in paths:
        sim = load_sim_for_plot(path, hide_stdout=True)
        try:
            values.append(get_values(sim))
        except ValueError:
            pass

    df = DataFrame(values, columns=columns)
    df["min_R"] = np.array([df.R2, df.R4]).min(axis=0)

    return df
