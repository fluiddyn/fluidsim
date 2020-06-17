import numpy as np

from util_submit import submit_profile

nxs_base = np.array([216, 160, 144]) // 2


def submit_profiles_1coef(coef):

    nxs = coef * nxs_base

    t_end = 0.5 / coef

    nx = nxs[0]
    coef_dealiasing = 2 / 3

    for scheme in ["RK4", "RK2"]:
        submit_profile(coef_dealiasing, nx, scheme, t_end=t_end)

    coefficients = [0.9, 1.0]

    for nx, coef_dealiasing in zip(nxs[1:], coefficients):
        for scheme in ["RK2_phaseshift_random", "RK2_phaseshift_exact"]:
            submit_profile(coef_dealiasing, nx, scheme, t_end=t_end)


for coef in [1, 2, 4]:
    submit_profiles_1coef(coef)
