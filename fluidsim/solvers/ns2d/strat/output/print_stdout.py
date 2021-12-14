"""Simple text output (:mod:`fluidsim.solvers.ns2d.strat.output.print_stdout`)
==============================================================================

.. autoclass:: PrintStdOutNS2DStrat
   :members:
   :private-members:

"""

import os

import numpy as np

from fluidsim.base.output.print_stdout import PrintStdOutBase

from fluiddyn.util import get_memory_usage
from fluiddyn.util import mpi


class PrintStdOutNS2DStrat(PrintStdOutBase):
    """Simple text output.

    Used to print in both the stdout and the stdout.txt file, and also
    to print simple info on the current state of the simulation.

    """

    def __init__(self, output):
        super().__init__(output)
        self.path_memory = self.output.path_run + "/memory_out.txt"
        if mpi.rank == 0 and self.output._has_to_save:
            if not os.path.exists(self.path_memory):
                self.file_memory = open(self.path_memory, "w")
            else:
                self.file_memory = open(self.path_memory, "r+")
                self.file_memory.seek(0, 2)  # go to the end of the file

    def _make_str_info(self):
        to_print = super()._make_str_info()

        energyK, energyA, energyK_ux = self.output.compute_energies()
        energy = energyK + energyA
        if mpi.rank == 0:
            to_print += (
                "              energyK = {:9.3e}\n"
                "              energyA = {:9.3e}\n"
                "              energy  = {:9.3e} ; Delta energy = {:+9.3e}\n"
                "".format(energyK, energyA, energy, energy - self.energy_temp)
            )

            if self.output._has_to_save:
                memory = get_memory_usage()
                self._write_memory_txt()

                to_print += f"              memory  = {memory:9.3f} Mo.\n"

            duration_left = self._evaluate_duration_left()
            if duration_left is not None:
                to_print += f"              estimated remaining duration = {duration_left}"

        self.energy_temp = energy
        return to_print

    def _write_memory_txt(self):
        """Write memory .txt"""
        it = self.sim.time_stepping.it
        mem = get_memory_usage()
        self.file_memory.write(f"{it:.3f},{mem:.3f}\n")
        self.file_memory.flush()
        os.fsync(self.file_memory.fileno())

    def plot_memory(self):
        """Plot memory used from memory_out.txt"""
        with open(self.output.path_run + "/memory_out.txt") as file_memory:
            lines = file_memory.readlines()

        lines_it = []
        lines_memory = []
        for il, line in enumerate(lines):
            lines_it.append(float(line.split(",")[0]))
            lines_memory.append(float(line.split(",")[1]))

        fig, ax = self.output.figure_axe()
        ax.set_xlabel("it")
        ax.set_ylabel("Memory (Mo)")

        ax.plot(lines_it, lines_memory, "k", linewidth=2)
        return fig

    def load(self):
        dict_results = {"name_solver": self.output.name_solver}

        with open(self.output.path_run + "/stdout.txt") as file_means:
            lines = file_means.readlines()

        lines_t = []
        lines_E = []
        lines_EK = []
        lines_EA = []
        for il, line in enumerate(lines):
            if line.startswith("it ="):
                lines_t.append(line)
            if line.startswith("              energy  ="):
                lines_E.append(line)
            if line.startswith("              energyK ="):
                lines_EK.append(line)
            if line.startswith("              energyA ="):
                lines_EA.append(line)

        nt = len(lines_t)
        if nt > 1:
            nt -= 1

        it = np.zeros(nt, dtype=int)
        t = np.zeros(nt)
        deltat = np.zeros(nt)

        E = np.zeros(nt)
        deltaE = np.zeros(nt)
        EK = np.zeros(nt)
        EA = np.zeros(nt)

        for il in range(nt):
            line = lines_t[il]
            words = line.split()
            it[il] = int(words[2])
            t[il] = float(words[6])
            deltat[il] = float(words[10])

            line = lines_E[il]
            words = line.split()
            E[il] = float(words[2])
            deltaE[il] = float(words[7])

            line = lines_EK[il]
            words = line.split()
            EK[il] = float(words[2])

            line = lines_EA[il]
            words = line.split()
            EA[il] = float(words[2])

        dict_results["it"] = it
        dict_results["t"] = t
        dict_results["deltat"] = deltat
        dict_results["E"] = E
        dict_results["deltaE"] = deltaE
        dict_results["EK"] = EK
        dict_results["EA"] = EA

        return dict_results

    def plot(self):
        dict_results = self.load()

        t = dict_results["t"]
        deltat = dict_results["deltat"]
        E = dict_results["E"]
        #  deltaE = dict_results['deltaE']
        EK = dict_results["EK"]
        EA = dict_results["EA"]

        x_left_axe = 0.12
        z_bottom_axe = 0.55
        width_axe = 0.85
        height_axe = 0.4
        size_axe = [x_left_axe, z_bottom_axe, width_axe, height_axe]
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel("t")
        ax1.set_ylabel("deltat(t)")

        ax1.set_title("info stdout\n" + self.output.summary_simul)
        ax1.plot(t, deltat, "k", linewidth=2)

        size_axe[1] = 0.08
        ax2 = fig.add_axes(size_axe)
        ax2.set_xlabel("t")
        ax2.set_ylabel("E(t), deltaE(t)")
        ax2.plot(t, E, "k", linewidth=2, label="$E$")
        # ax2.plot(t, deltaE, 'b', linewidth=2)
        ax2.plot(t, EK, "r", linewidth=2, label="$E_K$")
        ax2.plot(t, EA, "g", linewidth=2, label="$E_A$")
        ax2.legend()
        ax2.grid(True)

    def close(self):
        super().close()
        try:
            self.file_memory.close()
        except AttributeError:
            pass
