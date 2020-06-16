from pathlib import Path

import matplotlib.pyplot as plt


def get_params(path):
    name = path.name
    scheme, tmp = name.split("ns3d_")[1].split("_trunc")
    coef_dealiased, nx = tmp.split("x", 1)[0].split("_")
    coef_dealiased = float(coef_dealiased)
    nx = int(nx)
    if coef_dealiased == 0.667:
        coef_dealiased = 2 / 3
    return scheme, nx, coef_dealiased


def get_duration(path):
    for line in open(path):
        if "elapsed time" in line:
            return float(line.split()[3])


def get_values(path):
    values = list(get_params(path))
    values.append(get_duration(path))
    return values


columns = ["scheme", "nx", "coef_dealiased", "duration"]


def get_dataframe(path_dir):

    paths_log = sorted(path_dir.glob("*.log"))

    values = []
    for path in paths_log:
        values.append(get_values(path))

    df = DataFrame(values, columns=columns)
    df["nx2/3"] = df.nx * df.coef_dealiased / (2 / 3)
    # norm = df[df.scheme == "RK4"].duration.values[0]
    # df["speedup"] = norm / df["duration"]
    return df
