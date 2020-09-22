"""
check_freq_spectra_with_linear_mode.py
======================================
It performs a simulation with a initialization with a linear mode ap_fft.

It checks if the peak of the frequency spectra corresponds to the theoretical
frequency.

To compute the check:
---------------------
In sequential:
python check_freq_spectra_with_linear_mode.py

In MPI: (2 proc.)
mpirun -np 2 python check_freq_spectra_with_linear_mode.py

In MPI: (4 proc.)
mpirun -np 4 python check_freq_spectra_with_linear_mode.py
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from math import pi
from glob import glob
from fluidsim.solvers.ns2d.strat.solver import Simul


def _create_object_params():
    params = Simul.create_default_params()
    try:
        params.N = 1.0
    except AttributeError:
        pass

    # Operator parameters
    params.oper.nx = params.oper.ny = 32
    params.oper.Lx = params.oper.Ly = 2 * pi

    # Forcing parameters
    params.forcing.enable = False
    params.forcing.type = "tcrandom_anisotropic"

    try:
        params.forcing.tcrandom_anisotropic.angle = "45.0°"
    except AttributeError:
        pass

    params.forcing.nkmin_forcing = 8
    params.forcing.nkmax_forcing = 12

    # Compute \omega_l
    from math import radians

    if "°" in params.forcing.tcrandom_anisotropic.angle:
        angle = params.forcing.tcrandom_anisotropic.angle.split("°")
        angle = float(angle[0])
    else:
        raise ValueError("Angle should be contain the degrees symbol °.")
    omega_l = params.N * np.sin(radians(angle))
    params.forcing.tcrandom.time_correlation = 2 * pi / omega_l

    params.forcing.key_forced = "ap_fft"

    # Time stepping parameters
    params.time_stepping.USE_CFL = True
    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = 2.0

    # Output parameters
    params.output.HAS_TO_SAVE = False
    params.output.sub_directory = "tests"

    return params


if __name__ == "__main__":
    SAVE = True
    format = ".pdf"
    ### SHORT SIMULATION ###
    params = _create_object_params()

    params.oper.nx = nx = 128
    params.oper.ny = ny = nx // 4

    params.oper.Lx = 2 * pi
    params.oper.Ly = params.oper.Lx * (ny / nx)

    params.oper.NO_SHEAR_MODES = False
    params.nu_8 = 0.0
    params.N = 50.0

    params.time_stepping.USE_CFL = False
    params.time_stepping.deltat0 = 0.005
    params.time_stepping.t_end = 10.0
    params.time_stepping.cfl_coef_group = None

    params.output.HAS_TO_SAVE = True
    params.output.periods_print.print_stdout = 1.0
    params.output.periods_save.phys_fields = 2e-1

    params.output.periods_save.spatial_means = 0.0005

    params.output.periods_save.frequency_spectra = 1
    params.output.frequency_spectra.time_start = 0.0
    params.output.frequency_spectra.spatial_decimate = 1
    params.output.frequency_spectra.size_max_file = 10
    params.output.frequency_spectra.time_decimate = 4

    # Field initialization in the script
    params.init_fields.type = "linear_mode"
    params.init_fields.linear_mode.eigenmode = "ap_fft"
    params.init_fields.linear_mode.i_mode = (4, 1)
    params.init_fields.linear_mode.delta_k_adim = 1

    sim = Simul(params)
    sim.time_stepping.start()

    from fluiddyn.util import mpi

    if mpi.rank == 0:
        kx_s = sim.oper.KX[params.init_fields.linear_mode.i_mode]
        kz_s = sim.oper.KY[params.init_fields.linear_mode.i_mode]

        from math import pi

        omega_n = params.N * np.sin(
            np.arctan(
                sim.oper.kx[params.init_fields.linear_mode.i_mode[0]]
                / sim.oper.ky[params.init_fields.linear_mode.i_mode[1]]
            )
        )

        omega_n = omega_n / (2 * pi)

        ### COMPUTE FREQUENCY SPECTRA ###
        sim.output.frequency_spectra.compute_frequency_spectra()

        ### LOAD DATA AND PLOT ###
        path_file = glob(
            os.path.join(sim.output.path_run, "temporal_data", "temp_*")
        )[0]

        with h5py.File(path_file, "r") as f:
            omegas = f["omegas"][...]
            freq_spectrum = f["freq_spectrum"][...]

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlabel(r"$\omega / \omega_{th}$", fontsize=16)
        ax.set_ylabel(r"F($\omega$)", fontsize=16)

        # # For 10 conductivity probes
        # for i in range(0,10):
        #     ax.loglog(omegas/omega_n, freq_spectrum[0, :, 1, i])

        ax.semilogy(omegas / omega_n, freq_spectrum[0, :, 1, 4])
        ax.axvline(x=omega_n / omega_n, label=r"$\omega_{th}$", c="k")

        # Set text
        ax.text(
            5e-2,
            1e3,
            r"$\omega_{th} = N \sin(arctan \left( \frac{k_x}{k_z} \right))$",
            fontsize=16,
        )

        # If SAVE:
        if SAVE:
            path_root_save = "/home/users/calpelin7m/Phd/docs/Manuscript/figures"

            path_save = path_root_save + f"/test_frequency_spectra_seq{format}"
            if mpi.nb_proc > 1:
                path_save = (
                    path_root_save
                    + f"/test_frequency_spectra_mpi_{mpi.nb_proc}{format}"
                )
            fig.savefig(path_save, format="pdf")

        ax.legend()
        plt.show()
