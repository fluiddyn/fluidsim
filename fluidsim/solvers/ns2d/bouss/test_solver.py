import unittest

import matplotlib.pyplot as plt

import fluidsim as fls

import fluiddyn.util.mpi as mpi

from fluidsim.solvers.ns2d.bouss.solver import Simul
from fluidsim.solvers.ns2d.test_solver import TestSimulBase as Base


class TestSimulBase(Base):
    Simul = Simul


class TestForcingOutput(TestSimulBase):
    @classmethod
    def init_params(self):

        params = super().init_params()
        params.forcing.enable = True
        params.forcing.type = "tcrandom"

        # save all outputs!
        periods = params.output.periods_save
        for key in periods._key_attribs:
            periods[key] = 0.2

        params.output.ONLINE_PLOT_OK = True
        params.output.periods_print.print_stdout = 0.2
        params.output.periods_plot.phys_fields = 0.2

        params.output.increments.HAS_TO_PLOT_SAVED = True

        for tag in params.output._tag_children:
            if tag.startswith("periods"):
                continue
            child = params.output[tag]
            if hasattr(child, "HAS_TO_PLOT_SAVED"):
                child["HAS_TO_PLOT_SAVED"] = True

    def test_forcing_output(self):

        sim = self.sim

        sim.time_stepping.start()

        sim.state.compute("rot_fft")
        sim.state.compute("rot_fft")

        sim.state.compute("ux_fft")
        sim.state.compute("uy_fft")

        if mpi.nb_proc == 1:
            plt.close("all")
            sim.output.spectra.plot1d()
            sim.output.spectra.plot2d()
            sim.output.spectra.load2d_mean()
            sim.output.spectra.load1d_mean()

            sim.output.spatial_means.plot()
            sim.output.spatial_means.plot_dt_energy()
            sim.output.spatial_means.plot_dt_enstrophy()
            sim.output.spatial_means.compute_time_means()
            sim.output.spatial_means.load_dataset()
            sim.output.spatial_means.time_first_saved()
            sim.output.spatial_means.time_last_saved()

            sim.output.print_stdout.plot_energy()
            sim.output.print_stdout.plot_deltat()

            sim.output.spectra_multidim.plot()

            sim.output.spect_energy_budg.plot()
            with self.assertRaises(ValueError):
                sim.state.get_var("test")

            sim2 = fls.load_sim_for_plot(sim.output.path_run)
            sim2.output

            sim2.output.increments.load()
            sim2.output.increments.plot()
            sim2.output.increments.load_pdf_from_file()
            sim2.output.increments.plot_pdf()

            sim2.output.phys_fields.animate(
                "ux",
                dt_frame_in_sec=1e-6,
                dt_equations=0.3,
                repeat=False,
                clim=(-1, 1),
                save_file=False,
                numfig=1,
            )
            sim2.output.phys_fields.plot()

            # `compute('q')` two times for better coverage...
            sim.state.get_var("q")
            sim.state.get_var("q")
            sim.state.get_var("div")

            path_run = sim.output.path_run
            if mpi.nb_proc > 1:
                path_run = mpi.comm.bcast(path_run)

            sim3 = fls.load_state_phys_file(path_run, modif_save_params=False)
            sim3.params.time_stepping.t_end += 0.2
            sim3.time_stepping.start()

        plt.close("all")


if __name__ == "__main__":
    unittest.main()
