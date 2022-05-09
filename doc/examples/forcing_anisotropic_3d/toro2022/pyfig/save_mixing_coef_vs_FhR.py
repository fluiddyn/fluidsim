from util import save_fig, plt

from util_dataframe import plot, df

ax = plot(
    df,
    "Fh",
    "R2",
    c=df["Gamma"],
    logy=True,
    vmin=0.2,
    vmax=0.5,
    s=35,
)
ax.set_xlim(right=1)
ax.set_ylim(top=1e3)
ax.set_xlabel("$F_h$")
ax.set_ylabel(r"$\mathcal{R} = Re {F_h}^2$")

fig = ax.figure
fig.tight_layout()
fig.text(0.85, 0.07, r"$\Gamma=\frac{\epsilon_A}{\epsilon_K}$", fontsize=12)

save_fig(fig, "fig_mixing_coef_vs_FhR.png")

if __name__ == "__main__":
    plt.show()
