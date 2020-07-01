from util_submit import submit_simul

Re = 2800

nb_proc = 4
nx = 256
submit_simul(2 / 3, nx, "RK4", nb_proc=nb_proc, Re=Re)

nx = 168
submit_simul(1.0, nx, "RK4", nb_proc=nb_proc, Re=Re)
submit_simul(1.0, nx, "RK2_phaseshift_random", nb_proc=nb_proc, Re=Re)
submit_simul(1.0, nx, "RK2_phaseshift_exact", nb_proc=nb_proc, Re=Re)
