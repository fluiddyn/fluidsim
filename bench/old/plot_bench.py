from glob import glob
import json

import pandas as pd
import matplotlib.pyplot as plt

key_solver = "ns2d"
hostname = "cl7"

dicts = []

for path in glob("results_bench/*"):
    with open(path) as file:
        d = json.load(file)
    if not d["hostname"].startswith(hostname):
        continue
    dicts.append(d)


df = pd.DataFrame(dicts)


def plot_bench(nb_proc0=1):
    df0 = df.loc[df["nb_proc"] == nb_proc0]
    t_elapsed0 = df0["t_elapsed"].mean()
    times = df["t_elapsed"]

    plt.figure()
    ax = plt.subplot()
    ax.plot(df["nb_proc"], nb_proc0 * t_elapsed0 / times, "xr")
    tmp = [nb_proc0, df["nb_proc"].max()]
    ax.plot(tmp, tmp, "b-")
    ax.set_title("speed up")


plot_bench(nb_proc0=1)
plot_bench(nb_proc0=2)


plt.show()
