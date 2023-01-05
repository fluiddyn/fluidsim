"""Helper to plot remaining clock time data

Provides:

.. autoclass:: RemainingClockTime
   :members:
   :private-members:
   :undoc-members:

"""
from abc import ABCMeta, abstractmethod
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt


class RemainingClockTime(metaclass=ABCMeta):
    @abstractmethod
    def _load_times(self):
        """Load remaining time data."""

    def plot_clock_times(self):
        """Plot the estimated full clock time and clock time per time step."""

        results = self._load_times()
        equation_times = results["equation_times"]

        if len(equation_times) == 0:
            print("No time data in the log file. Can't plot anything.")
            return

        fig, axes = plt.subplots(2, 1, sharex=True)

        remaining_clock_times = results["remaining_clock_times"]

        ax = axes[0]
        ax.plot(equation_times, remaining_clock_times)
        ax.set_ylabel("estimated full clock time (s)")

        if remaining_clock_times[-1] > 0.05 * np.nanmax(remaining_clock_times):
            last_remaining = remaining_clock_times[-1]
            ax.plot(
                equation_times[-1],
                last_remaining,
                "rx",
                label=(
                    f"Last estimation {timedelta(seconds=int(last_remaining))}"
                ),
            )
            ax.legend()

        clock_times_per_timestep = results["clock_times_per_timestep"]
        if clock_times_per_timestep[-1] <= 0.0:
            equation_times = equation_times[:-1]
            clock_times_per_timestep = clock_times_per_timestep[:-1]

        times2 = np.empty(2 * equation_times.size)

        times2[0] = results["equation_time_start"]
        times2[1::2] = equation_times
        times2[2::2] = equation_times[:-1]

        clock_times_per_timestep2 = np.zeros_like(times2)
        clock_times_per_timestep2[::2] = clock_times_per_timestep
        clock_times_per_timestep2[1::2] = clock_times_per_timestep

        ax = axes[1]
        ax.plot(times2, clock_times_per_timestep2)
        ax.set_xlabel("equation time")
        ax.set_ylabel("clock time per time step (s)")
        full_clock_time = results["full_clock_time"]
        ax.set_title(
            f"Full clock time: {timedelta(seconds=int(full_clock_time))}"
        )

        for ax in axes:
            ax.set_ylim(bottom=0)

        fig.suptitle(self.output.summary_simul, fontsize=8)
        fig.tight_layout()

        print(
            "Mean clock time per time step: "
            f"{np.nanmean(clock_times_per_timestep2):.3g} s"
        )
