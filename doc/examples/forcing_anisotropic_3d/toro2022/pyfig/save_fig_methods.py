import matplotlib.pyplot as plt

from util import save_fig, get_paths_couple, get_customized_dataframe

N = 40
Rb = 20

paths = get_paths_couple(N, Rb)

print([p.name for p in paths])

df = get_customized_dataframe(paths)

print(df)

fig, axes = plt.subplots(nrows=2, sharex=True)

ax0, ax1 = axes

ax0.plot(df.nx, df["k_max*eta"], "o:", label=r"$k_{max} \eta$")
ax0.plot(
    df.nx,
    df["epsK2/epsK"],
    "s:",
    label=r"$\varepsilon_{K2} / (\varepsilon_{K2} + \varepsilon_{K4})$",
)

ax0.set_title(
    rf"$N = {N}$, "
    + r"$\mathcal{R}_i"
    + rf" = {Rb}$, $Re = {int(df.Re.max())}$ ($F_h = {df.Fh.mean():.2f}$, "
    r"$\mathcal{R}_2 = " + rf"{df.R2.mean():.1f}$)"
)

ax0.set_ylim([0, 1])

ax0.legend()
ax1.set_xlabel("$n_h$")

ax1.plot(
    df.nx, df["Gamma"], "o:", label=r"$\Gamma = \varepsilon_A / \varepsilon_K$"
)
ax1.plot(df.nx, df["I_velocity"], "s:", label=r"$I_{velo}$")
ax1.plot(df.nx, df["I_dissipation"], "^:", label=r"$I_{diss}$")
ax1.legend()


fig.tight_layout()
save_fig(fig, "fig_methods.png")

if __name__ == "__main__":
    plt.show()
