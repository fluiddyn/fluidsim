import fluiddyn as fld
from fluidsim.solvers.ns2d.solver import Simul

# Reusing the same parameters to modify
from simul_ns2d_plot import params


params.time_stepping.t_end = 10.

params.output.periods_save.phys_fields = 0.
params.output.periods_save.spectra = 0.
params.output.periods_save.spatial_means = 0.
params.output.periods_save.spect_energy_budg = 0.
params.output.periods_save.increments = 0.

params.output.periods_plot.phys_fields = 1.0

params.output.ONLINE_PLOT_OK = True

params.output.spectra.HAS_TO_PLOT_SAVED = False
params.output.spatial_means.HAS_TO_PLOT_SAVED = False
params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = False
params.output.increments.HAS_TO_PLOT_SAVED = False


if __name__ == '__main__':
    sim = Simul(params)
    sim.time_stepping.start()
    fld.show()
