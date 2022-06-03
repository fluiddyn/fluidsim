import numpy as np
import matplotlib.pyplot as plt

from util import save_fig, plot

from util_dataframe import df

ax = plot(
    df, "Fh", "I_velocity", c=np.log10(df["R2"]), vmin=0.5, vmax=2, logy=True
)

ax.set_xlabel("$F_h$")
ax.set_ylabel(r"$I_{velocity}$")

xs = np.linspace(1e-2, 1e-1, 4)
ax.plot(xs, 8e0 * xs**1)
ax.text(0.03, 0.4, "${F_h}^1$")

# ax.plot(xs, 8e1 * xs**2)
# ax.text(0.07, 0.2, "$F_h^{2}$")

fig = ax.figure

fig.text(0.84, 0.07, r"$\log_{10}(\mathcal{R})$", fontsize=12)

fig.tight_layout()
save_fig(fig, "fig_isotropy_velo_vs_Fh.png")

if __name__ == "__main__":
    plt.show()
