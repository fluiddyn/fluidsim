from util_submit import submit_simul

# Re = 2800
Re = None

nb_proc = 10
nx = 400
submit_simul(2 / 3, nx, "RK4", nb_proc=nb_proc, Re=Re)

nx = 280
submit_simul(1.0, nx, "RK4", nb_proc=nb_proc, Re=Re)
submit_simul(1.0, nx, "RK2_phaseshift_random", nb_proc=nb_proc, Re=Re)
submit_simul(1.0, nx, "RK2_phaseshift_exact", nb_proc=nb_proc, Re=Re)
