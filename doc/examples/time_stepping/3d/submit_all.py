from which_params import compute, coefficients, nx_bigs, nb_procs
from util_submit import submit_simul

schemes = [
    "RK4",
    "RK2",
    "RK2_phaseshift_exact",
    "RK2_phaseshift_random",
]


def submit(nx_big, nb_proc):
    nb_points = compute(nx_big, verbose=False)

    assert nb_points[0] == nx_big

    for coef_dealiasing, nx in zip(coefficients, nb_points):
        if nx == nx_big:
            schemes_nx = schemes[:2]
        else:
            schemes_nx = schemes

        for scheme in schemes_nx:
            if isinstance(scheme, tuple):
                scheme, cfl_coef = scheme
            else:
                cfl_coef = None
            print(f"{coef_dealiasing:.3f}, {nx}, {scheme:25s}, {cfl_coef}")
            submit_simul(coef_dealiasing, nx, scheme, cfl_coef, nb_proc)


for nx_big, nb_proc in zip(nx_bigs, nb_procs):
    if nx_big > 400:
        continue
    submit(nx_big, nb_proc)
