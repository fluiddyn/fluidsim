from run_simul import params, init_simul, run_simul

nx = 192 * 2
params.oper.nx = params.oper.ny = params.oper.nz = nx

params.oper.coef_dealiasing = 0.9

params.time_stepping.type_time_scheme = "RK2_phaseshift"
params.time_stepping.cfl_coef = 0.2

params.short_name_type_run = (
    f"clf{params.time_stepping.cfl_coef}_"
    f"coef_dealiasing{params.oper.coef_dealiasing}"
)

sim = init_simul(params)
run_simul(sim)
