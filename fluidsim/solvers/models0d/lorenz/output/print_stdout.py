"""Text output (:mod:`fluidsim.solvers.models0d.lorenz.output.print_stdout`)
============================================================================

.. autoclass:: PrintStdOutLorenz
   :members:
   :private-members:

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fluidsim.base.output.print_stdout import PrintStdOutBase

from fluiddyn.util import mpi


class PrintStdOutLorenz(PrintStdOutBase):
    """Simple text output.

    Used to print in both the stdout and the stdout.txt file, and also
    to print simple info on the current state of the simulation.

    """

    def complete_init_with_state(self):

        if self.period_print == 0:
            return

        self.t_last_print_info = -self.period_print

        self.print_stdout = self.__call__

    def _make_str_info(self):
        to_print = super()._make_str_info()

        if mpi.rank == 0:
            to_print += (
                (" " * 14) + "X = {:9.3e} ; Y = {:+9.3e} ; Z = {:+9.3e}\n" + "\n"
            ).format(
                float(self.sim.state.state_phys.get_var("X")),
                float(self.sim.state.state_phys.get_var("Y")),
                float(self.sim.state.state_phys.get_var("Z")),
            )

            duration_left = self._evaluate_duration_left()
            if duration_left is not None:
                to_print += f"              estimated remaining duration = {duration_left}"

        return to_print

    def load(self):
        dict_results = {"name_solver": self.output.name_solver}
        with open(self.output.path_run + "/stdout.txt") as file_means:
            lines = file_means.readlines()

        lines_t = []
        lines_X = []
        for il, line in enumerate(lines):
            if line.startswith("it ="):
                lines_t.append(line)
            if line.startswith(" " * 14 + "X ="):
                lines_X.append(line)

        nt = len(lines_t)
        if nt > 1:
            nt -= 1

        it = np.zeros(nt, dtype=int)
        t = np.zeros(nt)
        deltat = np.zeros(nt)

        X = np.zeros(nt)
        Y = np.zeros(nt)
        Z = np.zeros(nt)

        for il in range(nt):
            line = lines_t[il]
            words = line.split()
            it[il] = int(words[2])
            t[il] = float(words[6])
            deltat[il] = float(words[10])

            line = lines_X[il]
            words = line.split()
            X[il] = float(words[2])
            Y[il] = float(words[6])
            Z[il] = float(words[10])

        dict_results["it"] = it
        dict_results["t"] = t
        dict_results["deltat"] = deltat
        dict_results["X"] = X
        dict_results["Y"] = Y
        dict_results["Z"] = Z

        return dict_results

    def plot_deltat(self):
        dict_results = self.load()

        t = dict_results["t"]
        deltat = dict_results["deltat"]

        size_axe = [0.12, 0.12, 0.8, 0.8]
        fig, ax = self.output.figure_axe(size_axe=size_axe)

        ax.set_xlabel("t")
        ax.set_ylabel("deltat(t)")

        ax.set_title("info stdout\n" + self.output.summary_simul)
        ax.plot(t, deltat, "k", linewidth=2)

    def plot_XY_vs_time(self):
        dict_results = self.load()
        t = dict_results["t"]
        X = dict_results["X"]
        Y = dict_results["Y"]

        size_axe = [0.12, 0.12, 0.8, 0.8]
        fig, ax = self.output.figure_axe(size_axe=size_axe)

        ax.set_xlabel("$t$")
        ax.set_ylabel("$X$, $Y$")

        ax.plot(t, X, "b", label="$X$")
        ax.plot(t, Y, "r", label="$Y$")

        ax.plot(ax.get_xlim(), [self.sim.Xs0] * 2, "b--")
        ax.plot(ax.get_xlim(), [self.sim.Ys0] * 2, "r--")

        ax.plot(ax.get_xlim(), [self.sim.Xs1] * 2, "b.-")
        ax.plot(ax.get_xlim(), [self.sim.Ys1] * 2, "r.-")

        ax.legend()

    def plot_XY(self):
        dict_results = self.load()
        X = dict_results["X"]
        Y = dict_results["Y"]

        size_axe = [0.12, 0.12, 0.8, 0.8]
        fig, ax = self.output.figure_axe(size_axe=size_axe)

        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")

        ax.plot(X, Y, "b")

        ax.plot(self.sim.Xs0, self.sim.Ys0, "bx")

        ax.plot(self.sim.Xs1, self.sim.Ys1, "bx")

    def plot_XZ(self):
        dict_results = self.load()
        X = dict_results["X"]
        Z = dict_results["Z"]

        size_axe = [0.12, 0.12, 0.8, 0.8]
        fig, ax = self.output.figure_axe(size_axe=size_axe)

        ax.set_xlabel("$X$")
        ax.set_ylabel("$Z$")

        ax.plot(X, Z, "b")

        ax.plot(self.sim.Xs0, self.sim.Zs0, "bx")

        ax.plot(self.sim.Xs1, self.sim.Zs1, "bx")

    def plot_XYZ(self):
        dict_results = self.load()
        X = dict_results["X"]
        Y = dict_results["Y"]
        Z = dict_results["Z"]

        fig = plt.figure()
        size_axe = [0.12, 0.12, 0.8, 0.8]

        ax = fig.add_axes(size_axe, projection="3d")

        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
        ax.set_zlabel("$Z$")

        ax.plot(X, Y, Z, "b")

        ax.plot([self.sim.Xs0], [self.sim.Ys0], self.sim.Zs0, "bx")
        ax.plot([self.sim.Xs1], [self.sim.Ys1], self.sim.Zs1, "bx")
