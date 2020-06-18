from pathlib import Path

import pandas as pd
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
    path_dir = Path(path_dir)
    paths_log = sorted(path_dir.glob("*.log"))

    values = []
    for path in paths_log:
        values.append(get_values(path))

    df_full = pd.DataFrame(values, columns=columns)
    df_full["nx2/3"] = df_full.nx * df_full.coef_dealiased / (2 / 3)

    nx23s = df_full["nx2/3"].unique()

    results = []
    for nx23 in nx23s:
        df = df_full[df_full["nx2/3"] == nx23].copy()
        try:
            norm = df[df.scheme == "RK4"].duration.values[0]
        except IndexError:
            # No data for RK4
            continue
        df["speedup"] = norm / df["duration"]
        results.append(df)
    return pd.concat(results)


df = get_dataframe(
    Path.home() / "Dev/postdoc_legi_jason/phaseshift/profiles_2020-06-17"
)

print(df)
schemes = sorted(df.scheme.unique())

fig, ax = plt.subplots()

for scheme in schemes:
    df_1scheme = df[df.scheme == scheme].copy()
    df_1scheme.sort_values("nx2/3", inplace=True)
    coefs = df_1scheme.coef_dealiased.unique()
    coefs.sort()
    for coef in coefs:

        df_tmp = df_1scheme[df_1scheme.coef_dealiased == coef]
        if coef == 2 / 3:
            str_coef = "2/3"
        else:
            str_coef = str(coef)
        ax.plot(
            df_tmp["nx2/3"], df_tmp.speedup, "x", label=f"{scheme}, {str_coef}"
        )

ax.set_xscale("log")
fig.legend()

plt.show()
