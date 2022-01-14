import numpy as np
import matplotlib.pyplot as plt

from fluidsim.base.output.print_stdout import PrintStdOutBase

from fluiddyn.util import mpi


class PrintStdOutNS3D(PrintStdOutBase):
    """Used to print in both the stdout and the stdout.txt file, and also
    to print simple info on the current state of the simulation.

    """

    def _make_str_info(self):
        to_print = super()._make_str_info()

        energy = self.output.compute_energy()
        if mpi.rank == 0:
            to_print += (
                "              energy = {:9.3e} ; Delta energy = {:+9.3e}\n"
                "".format(energy, energy - self.energy_temp)
            )

            duration_left = self._evaluate_duration_left()
            if duration_left is not None:
                to_print += f"              estimated remaining duration = {duration_left}"

        self.energy_temp = energy
        return to_print

    def load(self):
        dict_results = {"name_solver": self.output.name_solver}

        with open(self.output.path_run + "/stdout.txt") as file_means:
            lines = file_means.readlines()

        lines_t = []
        lines_E = []
        for il, line in enumerate(lines):
            if line[0:4] == "it =":
                lines_t.append(line)
            if line[0:22] == "              energy =":
                lines_E.append(line)

        nt = len(lines_t)
        if nt > 1:
            nt -= 1

        it = np.zeros(nt, dtype=int)
        t = np.zeros(nt)
        deltat = np.zeros(nt)

        E = np.zeros(nt)
        deltaE = np.zeros(nt)

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

        dict_results["it"] = it
        dict_results["t"] = t
        dict_results["deltat"] = deltat
        dict_results["E"] = E
        dict_results["deltaE"] = deltaE

        return dict_results

    def plot(self):
        dict_results = self.load()

        t = dict_results["t"]
        deltat = dict_results["deltat"]
        E = dict_results["E"]
        deltaE = dict_results["deltaE"]

        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.set_xlabel("t")
        ax1.set_ylabel("deltat(t)")

        ax1.set_title("info stdout\n" + self.output.summary_simul)
        ax1.plot(t, deltat, "k", linewidth=2)

        ax2.set_xlabel("t")
        ax2.set_ylabel("E(t), deltaE(t)")
        ax2.plot(t, E, "k", linewidth=2)
        ax2.plot(t, deltaE, "b", linewidth=2)

        fig.tight_layout()
