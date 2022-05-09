
import matplotlib.pyplot as plt

from fluidsim.util import get_dataframe_from_paths

from util import couples320, get_path_finer_resol


paths = []
for N, Rb in sorted(couples320):
    paths.append(get_path_finer_resol(N, Rb))


print(f"Using {len(paths)} simulations")


def customize(result, sim):
    result["Rb"] = float(sim.params.short_name_type_run.split("_Rb")[-1])
    result["nx"] = sim.params.oper.nx


df = get_dataframe_from_paths(
    paths, tmin="t_start+2", use_cache=1, customize=customize
)
df["Re"] = df.Rb * df.N**2

columns_old = df.columns.tolist()

# fmt: off
first_columns = [
    "N", "Rb", "Re", "nx", "Fh", "R2", "k_max*eta", "epsK2/epsK", "Gamma",
    "lx1", "lx2", "lz1", "lz2", "I_velocity", "I_dissipation"]
# fmt: on

columns = first_columns.copy()
for key in columns_old:
    if key not in columns:
        columns.append(key)

df = df[columns]


def plot(
    df,
    x,
    y,
    logx=True,
    logy=False,
    c=None,
    vmin=None,
    vmax=None,
    s=None,
):
    ax = df.plot.scatter(
        x=x,
        y=y,
        logx=logx,
        logy=logy,
        c=c,
        edgecolors="k",
        vmin=vmin,
        vmax=vmax,
        s=s,
    )
    pc = ax.collections[0]
    pc.set_cmap("inferno")
    plt.colorbar(pc, ax=ax)
    return ax
