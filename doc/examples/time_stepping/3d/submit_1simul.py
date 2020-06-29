from util_submit import submit_simul

nb_proc = 10
nx = 400

submit_simul(2 / 3, nx, "RK4", nb_proc=nb_proc, truncation_shape="cubic")
submit_simul(1.0, nx, "RK4", nb_proc=nb_proc, truncation_shape="cubic")
submit_simul(2 / 3, nx, "RK4", nb_proc=nb_proc)
submit_simul(1.0, nx, "RK4", nb_proc=nb_proc)

submit_simul(1.0, nx, "RK2_phaseshift_random", nb_proc=nb_proc)
submit_simul(1.0, nx, "RK2_phaseshift_exact", nb_proc=nb_proc)
