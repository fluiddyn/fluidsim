from util_submit import submit_profile

coef_dealiasing = 1
nx = 576
scheme = "RK2_phaseshift_random"

submit_profile(
    coef_dealiasing, nx, scheme, t_end=0.5 / 8, nb_pairs=1, nb_steps=None
)
