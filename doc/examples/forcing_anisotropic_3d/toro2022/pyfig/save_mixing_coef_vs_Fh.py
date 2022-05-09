import numpy as np
import matplotlib.pyplot as plt

from util import save_fig, plot

from util_dataframe import df

ax = plot(
    df,
    "Fh",
    "Gamma",
    c=np.log10(df["R2"]),
    logy=True,
    vmin=0.3,
    vmax=2,
    s=35,
)
# ax.set_xlim(right=1)
# ax.set_ylim(top=1e3)
ax.set_xlabel("$F_h$")
ax.set_ylabel(r"$\Gamma=\epsilon_A / \epsilon_K$")

xs = np.linspace(1.5e-1, 5e-1, 4)
ax.plot(xs, 5e-2 * xs**-1)
ax.text(0.16, 0.12, "${F_h}^{-1}$")

xs = np.linspace(6e-1, 4, 4)
ax.plot(xs, 5e-2 * xs**-2)
ax.text(1.2, 0.05, "${F_h}^{-2}$")

fig = ax.figure
fig.tight_layout()
fig.text(0.85, 0.07, r"$\mathcal{R}$", fontsize=12)

save_fig(fig, "fig_mixing_coef_vs_Fh.png")

if __name__ == "__main__":
    plt.show()
