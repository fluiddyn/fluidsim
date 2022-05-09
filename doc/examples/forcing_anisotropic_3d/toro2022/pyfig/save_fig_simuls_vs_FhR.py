import matplotlib.pyplot as plt

from util import save_fig, plot

from util_dataframe import df

ax = plot(df, "Fh", "Rb", logy=True)
ax.set_xlim(right=1)
ax.set_ylim(top=1e3)

ax.set_xlabel("$F_h$")
ax.set_ylabel("$\mathcal{R} = Re {F_h}^2$")

fig = ax.figure
fig.tight_layout()

save_fig(fig, "fig_simuls_vs_FhR.png")

if __name__ == "__main__":
    plt.show()
