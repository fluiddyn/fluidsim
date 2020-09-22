"""create new files with a modificated resolution."""

from solveq2d import solveq2d

t_approx = None
coef_modif_resol = 2
dir_base = "~/Storage/Results_SW1lw/Pure_standing_waves_1920x1920"

print("run the function solveq2d.modif_resolution_all_dir")

solveq2d.modif_resolution_all_dir(
    t_approx=t_approx, coef_modif_resol=coef_modif_resol, dir_base=dir_base
)
