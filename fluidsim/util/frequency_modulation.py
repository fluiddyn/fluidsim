import numpy as np

from fluiddyn.calcul.easypyfft import FFTW1DReal2Complex, fftw_grid_size


class FrequencyModulatedSignalMaker:
    def __init__(self, total_time, approximate_dt):
        self.nb_points = 2 * fftw_grid_size(int(total_time / approximate_dt))
        self.dt = total_time / self.nb_points
        self.times = self.dt * np.arange(self.nb_points)

        self.oper_fft = FFTW1DReal2Complex(self.nb_points)
        nb_omegas = self.oper_fft.shapeK[0]
        domega = 2 * np.pi / total_time
        self.omegas = domega * np.arange(nb_omegas)
        self.omegas_nozero = self.omegas.copy()
        self.omegas_nozero[0] = 1e-14

    def create_frequency_modulated_signal(
        self,
        omega_f,
        delta_omega_f,
        amplitude=1.0,
        omega_min=None,
        omega_max=None,
    ):
        # filter parameters for the time signal
        if omega_min is None:
            omega_min = omega_f / 10
        if omega_max is None:
            omega_max = omega_f
        # Gaussian white noise
        omegaf_vs_time = np.random.randn(self.nb_points)
        omegaf_vs_omega = self.oper_fft.fft(omegaf_vs_time)
        # rectangular filtering
        omegaf_vs_omega *= self.omegas >= omega_min
        omegaf_vs_omega *= self.omegas <= omega_max
        # set amplitude
        omegaf_vs_time = self.oper_fft.ifft(omegaf_vs_omega)
        omegaf_vs_time *= delta_omega_f / np.std(omegaf_vs_time)
        # time integration for frequency modulation
        omegaf_vs_omega = self.oper_fft.fft(omegaf_vs_time)
        phase_vs_omega = omegaf_vs_omega / (1j * self.omegas_nozero)
        phase_vs_time = self.oper_fft.ifft(phase_vs_omega)
        # frequency modulated forcing time signal
        phase_vs_time += omega_f * self.times - phase_vs_time[0]
        forcing_vs_time = (
            amplitude * (omegaf_vs_time + omega_f) * np.sin(phase_vs_time)
        )
        return self.times, forcing_vs_time
