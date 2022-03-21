from pathlib import Path
import os

import numpy as np
from pandas import DataFrame

from fluidsim import load_sim_for_plot


def get_isotropy_values(sim, t_statio, recompute=False):

    first_line = "# isotropy_velocity isotropy_dissipation t_last"

    t_last = sim.output.spatial_means.time_last_saved()

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

        return isotropy_velocity, isotropy_dissipation

    print(f"compute for simulation\n{sim.output.path_run}")
    isotropy_velocity = sim.output.spectra.compute_isotropy_velocities(t_statio)
    isotropy_dissipation = (
        sim.output.spect_energy_budg.compute_isotropy_dissipation(t_statio)
    )

    with open(path_file, "w") as file:
        file.write(
            f"{first_line}\n{isotropy_velocity} "
            f"{isotropy_dissipation} {t_last}\n"
        )

    return isotropy_velocity, isotropy_dissipation


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
        I_velocities, I_dissipation = get_isotropy_values(sim, t_statio)
    except OSError:
        print(sim.output.path_run)
        raise

    # fmt: off
    return [
        N, U2, epsK, Gamma, Fh, R2, R4,
        I_velocities, I_dissipation,
    ]
    # fmt: on


# fmt: off
columns = [
    "N",
    #"Rb", "Re",
    "U2", "epsK", "Gamma", "Fh", "R2", "R4",
    "I_velocities", "I_dissipation",
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
