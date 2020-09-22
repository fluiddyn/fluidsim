import matplotlib.pylab as plt

from solveq2d import solveq2d


name_dir = (
    "/scratch/augier/"
    "/Results_SW1lw"
    "/Pure_standing_waves_7680x7680"
    "/SE2D_SW1lwaves_forcingw_L=50.x50._7680x7680_c=40_f=0_2013-08-06_12-26-05"
)

tmin = 194
tmax = 1000


# sim = solveq2d.load_state_phys_file(name_dir)
# sim.output.phys_fields.plot(key_field='ux')
# sim.output.phys_fields.plot(key_field='eta')


sim = solveq2d.create_sim_plot_from_dir(name_dir=name_dir)

# sim.output.print_stdout.plot()
# sim.output.spatial_means.plot()

# sim.output.spectra.plot1D(tmin=tmin, tmax=tmax, delta_t=0.,
#                           coef_compensate=2.)

# sim.output.spectra.plot2D(tmin=tmin, tmax=tmax, delta_t=0.,
#                           coef_compensate=2.)

# sim.output.spect_energy_budg.plot(tmin=tmin, tmax=tmax, delta_t=0.)

# sim.output.increments.plot(tmin=tmin, tmax=None, delta_t=0.,
#                            order=4, yscale='log')

# sim.output.prob_dens_func.plot(tmin=tmin)

sim.output.time_sigK.plot_spectra()


solveq2d.show()
