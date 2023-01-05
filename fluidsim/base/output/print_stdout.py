"""Standard output saving
=========================

Provides:

.. autoclass:: PrintStdOutBase
   :members:
   :private-members:
   :noindex:
   :undoc-members:

"""

from time import time
import os
import sys
from datetime import timedelta

import numpy as np

from fluiddyn.util import mpi, print_memory_usage

from fluidsim_core.output.remaining_clock_time import RemainingClockTime

from fluidsim.util import times_start_last_from_path


class PrintStdOutBase(RemainingClockTime):
    """A :class:`PrintStdOutBase` object is used to print in both the
    stdout and the stdout.txt file, and also to print simple info on
    the current state of the simulation."""

    _tag = "print_stdout"

    @staticmethod
    def _complete_params_with_default(params):
        params.output.periods_print._set_attrib("print_stdout", 1.0)

    def __init__(self, output):
        sim = output.sim
        params = sim.params

        self.output = output
        self.sim = sim
        self.params = params

        try:
            self.c2 = params.c2
            self.f = params.f
            self.nx = params.oper.nx
        except AttributeError:
            pass

        self.period_print = params.output.periods_print.print_stdout

        self.path_file = self.output.path_run + "/stdout.txt"

        if mpi.rank == 0 and self.output._has_to_save:
            if not os.path.exists(self.path_file):
                self.file = open(self.path_file, "w")
            else:
                self.file = open(self.path_file, "r+")
                self.file.seek(0, 2)  # go to the end of the file

    def complete_init_with_state(self):

        self.energy0 = self.output.compute_energy()

        if self.period_print == 0:
            return

        self.energy_temp = self.energy0 + 0.0
        self.t_last_print_info = -self.period_print

        self.print_stdout = self.__call__

    def __call__(self, to_print, end="\n"):
        """Print in stdout and if SAVE in the file stdout.txt"""
        if mpi.rank == 0:
            print(to_print, end=end)
            sys.stdout.flush()
            if self.output._has_to_save:
                self.file.write(to_print + end)
                self.file.flush()
                os.fsync(self.file.fileno())

    def _online_print(self):
        """Print simple info on the current state of the simulation"""
        tsim = self.sim.time_stepping.t
        if (
            tsim + 1e-15
        ) // self.period_print > self.t_last_print_info // self.period_print:
            self._print_info()
            self.t_last_print_info = tsim

    def _print_info(self):
        self.print_stdout(self._make_str_info())
        print_memory_usage("MEMORY_USAGE")

    def _make_str_info(self):
        ts = self.sim.time_stepping
        return (
            f"it = {ts.it:6d} ; t = {ts.t:12.6g} ; deltat  = {ts.deltat:10.5g}\n"
        )

    def _evaluate_duration_left(self):
        """Computes the remaining time."""
        t_clock = time()
        try:
            delta_clock_time = t_clock - self.t_clock_last
        except AttributeError:
            self.t_clock_last = t_clock
            return

        self.t_clock_last = t_clock
        tsim = self.sim.time_stepping.t

        # variables delta_...: differences between 2 consecutive savings
        delta_equation_time = tsim - self.t_last_print_info

        if delta_equation_time == 0:
            return

        if self.params.time_stepping.USE_T_END:
            remaining_equation_time = self.params.time_stepping.t_end - tsim
        else:
            remaining_equation_time = (
                self.params.time_stepping.it_end - self.sim.time_stepping.it
            ) * self.sim.time_stepping.deltat

        if remaining_equation_time < 0:
            return

        remaining_clock_time = round(
            remaining_equation_time / delta_equation_time * delta_clock_time
        )
        return timedelta(seconds=remaining_clock_time)

    def close(self):
        try:
            self.file.close()
        except AttributeError:
            pass

    def _load_times(self):
        """Load time data from the log file"""
        equation_times = []
        time_steps = []
        remaining_clock_times = []
        times_end = []
        # variables delta_...: differences between 2 consecutive savings
        delta_time_inds = []
        delta_equation_times = []

        time_last = None
        time = None
        it_last = None
        it = None
        equation_time_start = None

        with open(self.output.path_run + "/stdout.txt") as file:
            for line in file:
                if line.startswith("it = "):
                    words = line.split()
                    time_last = time
                    it_last = it
                    it = int(words[2])
                    time = float(words[6])
                    delta_t = float(words[10])

                    if equation_time_start is None:
                        equation_time_start = time

                elif line.startswith("    compute until t ="):
                    t_end = float(line.split()[4])

                elif line.startswith(
                    "              estimated remaining duration"
                ):
                    words = line.split()
                    if "day" in line:
                        day = int(words[4])
                        hour_min_second = words[6]
                    else:
                        day = 0
                        hour_min_second = words[4]

                    nb_hours, nb_minutes, nb_seconds = tuple(
                        int(part) for part in hour_min_second.split(":")
                    )

                    delta = timedelta(
                        days=day,
                        hours=nb_hours,
                        minutes=nb_minutes,
                        seconds=nb_seconds,
                    )

                    remaining_clock_times.append(delta.total_seconds())
                    equation_times.append(time)
                    delta_equation_times.append(time - time_last)
                    delta_time_inds.append(it - it_last)
                    times_end.append(t_end)
                    time_steps.append(delta_t)

        remaining_clock_times = np.array(remaining_clock_times)
        equation_times = np.array(equation_times)
        delta_equation_times = np.array(delta_equation_times)
        times_end = np.array(times_end)
        time_steps = np.array(time_steps)
        delta_time_inds = np.array(delta_time_inds)

        remaining_equation_times = times_end - equation_times
        # see how remaining_clock_time is computed in _evaluate_duration_left
        delta_clock_times = (
            delta_equation_times
            * remaining_clock_times
            / remaining_equation_times
        )
        full_clock_time = delta_clock_times[np.isfinite(delta_clock_times)].sum()

        clock_times_per_timestep = delta_clock_times / delta_time_inds

        loc = locals()
        return {
            key: loc[key]
            for key in (
                "remaining_clock_times",
                "equation_times",
                "delta_equation_times",
                "times_end",
                "time_steps",
                "delta_time_inds",
                "clock_times_per_timestep",
                "equation_time_start",
                "delta_clock_times",
                "full_clock_time",
            )
        }

    def get_times_start_last(self):
        return times_start_last_from_path(self.sim.output.path_run)
