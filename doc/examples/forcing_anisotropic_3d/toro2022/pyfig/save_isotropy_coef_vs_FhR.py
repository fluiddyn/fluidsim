import numpy as np
import matplotlib.pyplot as plt

from util import save_fig, plot

from util_dataframe import df

ax = plot(
    df,
    "Fh",
    "R2",
    c=df["I_velocity"],
    logy=True,
    vmin=0.2,
    vmax=0.8,
    s=50 * df["I_dissipation"],
)
ax.set_xlim(right=1)
ax.set_ylim(top=1e3)
ax.set_xlabel("$F_h$")
ax.set_ylabel("$\mathcal{R} = Re {F_h}^2$")
fig = ax.figure
fig.tight_layout()

fig.text(0.84, 0.07, r"$I_\mathit{velocity}$", fontsize=12)

ax_legend = fig.add_axes([0.17, 0.76, 0.2, 0.16])
ax_legend.set_xticklabels([])
ax_legend.set_xticks([])
ax_legend.set_yticklabels([])
ax_legend.set_yticks([])
isotropy_diss = np.array([0.1, 0.5, 0.9])
heights = np.array([0.2, 0.5, 0.8])
ax_legend.scatter([0.15, 0.15, 0.15], heights, s=50 * isotropy_diss)
ax_legend.set_xlim([0, 1])
ax_legend.set_ylim([0, 1])

for h, i in zip(heights, isotropy_diss):
    ax_legend.text(0.28, h - 0.06, r"$I_\mathit{diss} = " + f"{i}$")


save_fig(fig, "fig_isotropy_coef_vs_FhR.png")

if __name__ == "__main__":
    plt.show()
