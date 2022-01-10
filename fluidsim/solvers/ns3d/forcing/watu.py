"""Forcing (:mod:`fluidsim.solvers.ns3d.forcing.watu`)
======================================================

.. autoclass:: ForcingInternalWavesWatuCoriolis
   :members:

"""

from pathlib import Path
from math import pi

import numpy as np
import h5netcdf
from scipy.interpolate import interp1d

from transonic import boost, Array

from fluiddyn.util import mpi
from fluidsim.util.frequency_modulation import FrequencyModulatedSignalMaker

from fluidsim.base.forcing.base import ForcingBasePseudoSpectral
from fluidsim.base.forcing.specific import SpecificForcingPseudoSpectralSimple


class ForcingInternalWavesWatuCoriolis(SpecificForcingPseudoSpectralSimple):
    """Forcing mimicking an experimental setup in the Coriolis platform.

    The experiments have been carried out within the ERC project `WATU
    <http://nicolas.mordant.free.fr/watu.html>`_. Internal gravity waves are
    forced with two long vertical boards oscillating along their central
    horizontal axes.

    """

    tag = "watu_coriolis"

    @classmethod
    def _complete_params_with_default(cls, params):
        params.forcing.available_types.append(cls.tag)
        params.forcing._set_child(
            cls.tag,
            dict(
                omega_f=0.3,  # rad/s
                delta_omega_f=0.03,  # rad/s
                amplitude=0.05,  # m
                period_forcing=1000,
                approximate_dt=None,
                nb_wave_makers=2,
            ),
        )

    @classmethod
    def _modify_sim_repr_maker(cls, sim_repr_maker):
        p_watu = sim_repr_maker.sim.params.forcing[cls.tag]
        sim_repr_maker.add_parameters(
            {"ampl": p_watu.amplitude, "omegaf": p_watu.omega_f}
        )

    def __init__(self, sim):
        super().__init__(sim)

        self.nb_wave_makers = sim.params.forcing.watu_coriolis.nb_wave_makers
        if self.nb_wave_makers not in (1, 2):
            raise NotImplementedError

        # preparation of a time signal for the forcing
        if mpi.rank != 0:
            self.interpolents = None
        else:
            path_file = Path(sim.output.path_run) / "forcing_watu_coriolis.h5"

            if path_file.exists():
                # load time signals
                with h5netcdf.File(str(path_file), "r") as file:

                    times = file["/times"][...]
                    signals = file["/signals"][...]

            else:
                time_signal_maker = FrequencyModulatedSignalMaker(
                    total_time=sim.params.forcing.watu_coriolis.period_forcing,
                    approximate_dt=sim.params.forcing.watu_coriolis.approximate_dt,
                )

                signals = []
                for _ in range(self.nb_wave_makers):
                    (
                        times,
                        forcing_vs_time,
                    ) = time_signal_maker.create_frequency_modulated_signal(
                        sim.params.forcing.watu_coriolis.omega_f,
                        sim.params.forcing.watu_coriolis.delta_omega_f,
                        sim.params.forcing.watu_coriolis.amplitude,
                    )
                    signals.append(forcing_vs_time)

                signals = np.array(signals)
                # save time signals
                with h5netcdf.File(str(path_file), "w") as file:
                    file.create_variable("/times", ("time",), data=times)
                    file.create_variable(
                        "/signals", ("signal", "time"), data=signals
                    )

            # add one point at the end for periodicity
            signals = np.column_stack((signals, signals[:, 0]))
            times = np.hstack(
                (times, sim.params.forcing.watu_coriolis.period_forcing)
            )

            # interpolation functions
            self.interpolents = [
                interp1d(times, signals[index])
                for index in range(signals.shape[0])
            ]

            # warning : period_forcing / omega_f should be integer
            period_f = 2 * pi / sim.params.forcing.watu_coriolis.omega_f
            if sim.params.forcing.watu_coriolis.period_forcing % period_f > 1e-10:
                mpi.printby0(
                    "WARNING: period_forcing is not a multiple of 2*pi/omega_f."
                    "The forcing velocity signal may not be continuous."
                )

        if mpi.nb_proc > 1:
            self.interpolents = mpi.comm.bcast(self.interpolents, root=0)

        amplitude = sim.params.forcing.watu_coriolis.amplitude
        lx = sim.params.oper.Lx
        ly = sim.params.oper.Ly
        lz = sim.params.oper.Lz
        dx = lx / sim.params.oper.nx

        # calculus of the target velocity components
        width = max(4 * dx, amplitude / 5)

        def step_func(x):
            """Activation function"""
            return 0.5 * (np.tanh(x / width) + 1)

        amplitude_side = amplitude + 0.15
        X, Y, Z = sim.oper.get_XYZ_loc()

        self.maskx = (
            (step_func(-(X - amplitude)) + step_func(X - (lx - amplitude)))
            * step_func(Y - amplitude_side)
            * step_func(-(Y - (ly - amplitude_side)))
        )

        self.masky = (
            (step_func(-(Y - amplitude)) + step_func(Y - (ly - amplitude)))
            * step_func(X - amplitude_side)
            * step_func(-(X - (lx - amplitude_side)))
        )

        z_variation = np.sin(2 * np.pi * Z / lz)
        self.vxtarget = z_variation
        self.vytarget = z_variation

        # calculus of coef_sigma

        # f(t) = f0 * exp(-sigma*t)
        # If we want f(t)/f0 = 10**(-gamma) after n_dt time steps, we have to have:
        # sigma = gamma / (n_dt * dt)

        gamma = 2
        n_dt = 4
        self.coef_sigma = gamma / n_dt

        self.sigma = self.coef_sigma / self.params.time_stepping.deltat_max
        self.period_forcing = self.params.forcing.watu_coriolis.period_forcing

        self._tmp_forcing = sim.oper.create_arrayX()

    def compute_forcing_fft_each_time(self):
        sim = self.sim
        time = sim.time_stepping.t % self.period_forcing
        coef_forcing_time_x = float(self.interpolents[0](time))
        vx = sim.state.state_phys.get_var("vx")
        tmp = self._tmp_forcing
        fx = compute_watu_coriolis_forcing_component(
            self.sigma, self.maskx, coef_forcing_time_x, self.vxtarget, vx, tmp
        )
        fx_fft = sim.oper.fft(fx)
        if self.nb_wave_makers == 1:
            return {"vx_fft": fx_fft}
        coef_forcing_time_y = float(self.interpolents[1](time))
        vy = sim.state.state_phys.get_var("vy")
        fy = compute_watu_coriolis_forcing_component(
            self.sigma, self.masky, coef_forcing_time_y, self.vytarget, vy, tmp
        )
        return {"vx_fft": fx_fft, "vy_fft": sim.oper.fft(fy)}


A2d = Array[np.float64, "3d"]
A3d = Array[np.float64, "3d"]


@boost
def compute_watu_coriolis_forcing_component(
    sigma: float,
    mask: A2d,
    coef_forcing_time: float,
    target: A3d,
    velocity: A3d,
    out: A3d,
):
    out[:] = sigma * mask * (coef_forcing_time * target - velocity)
    return out
