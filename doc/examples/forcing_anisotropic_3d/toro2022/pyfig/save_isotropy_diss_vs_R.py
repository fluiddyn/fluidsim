import numpy as np
import matplotlib.pyplot as plt

from util import save_fig, plot

from util_dataframe import df

ax = plot(df, "R2", "I_dissipation", c=np.log10(df["Fh"]), vmin=-2, vmax=-1)

ax.set_xlabel(r"$\mathcal{R}$")
ax.set_ylabel("$I_{diss}$")

fig = ax.figure

fig.text(0.84, 0.07, r"$\log_{10}(F_h)$", fontsize=12)

fig.tight_layout()
save_fig(fig, "fig_isotropy_diss_vs_R.png")

if __name__ == "__main__":
    plt.show()
