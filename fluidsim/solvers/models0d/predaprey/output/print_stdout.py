"""Simple text output (:mod:`fluidsim.solvers.ns2d.output.print_stdout`)
========================================================================

.. autoclass:: PrintStdOutPredaPrey
   :members:
   :private-members:

"""

from __future__ import print_function, division

from builtins import range
import numpy as np

from fluidsim.base.output.print_stdout import PrintStdOutBase

from fluiddyn.util import mpi


class PrintStdOutPredaPrey(PrintStdOutBase):
    """Simple text output.

    Used to print in both the stdout and the stdout.txt file, and also
    to print simple info on the current state of the simulation.

    """

    def complete_init_with_state(self):

        self.potential0 = self.output.compute_potential()

        if self.period_print == 0:
            return

        self.potential_tmp = self.potential0
        self.t_last_print_info = -self.period_print

        self.print_stdout = self.__call__

    def _make_str_info(self):
        to_print = super(PrintStdOutPredaPrey, self)._make_str_info()

        potential = self.output.compute_potential()
        if mpi.rank == 0:
            to_print += (
                (' ' * 14) + 'X = {:9.3e} ; Y = {:+9.3e}\n' +
                (' ' * 14) + 'potential = {:9.3e} ; Delta pot = {:+9.3e}'
                '\n').format(float(self.sim.state.state_phys.get_var('X')),
                             float(self.sim.state.state_phys.get_var('Y')),
                             potential, potential-self.potential_tmp)

            duration_left = self._evaluate_duration_left()
            if duration_left is not None:
                to_print += (
                    '              estimated remaining duration = {:9.3g} s'
                    ''.format(duration_left))

        self.potential_temp = potential
        return to_print

    def load(self):
        dico_results = {'name_solver': self.output.name_solver}
        with open(self.output.path_run + '/stdout.txt') as file_means:
            lines = file_means.readlines()

        lines_t = []
        lines_P = []
        lines_X = []
        for il, line in enumerate(lines):
            if line.startswith('it ='):
                lines_t.append(line)
            if line.startswith(' ' * 14 + 'potential ='):
                lines_P.append(line)
            if line.startswith(' ' * 14 + 'X ='):
                lines_X.append(line)

        nt = len(lines_t)
        if nt > 1:
            nt -= 1

        it = np.zeros(nt, dtype=np.int)
        t = np.zeros(nt)
        deltat = np.zeros(nt)

        P = np.zeros(nt)
        deltaP = np.zeros(nt)

        X = np.zeros(nt)
        Y = np.zeros(nt)

        for il in range(nt):
            line = lines_t[il]
            words = line.split()
            it[il] = int(words[2])
            t[il] = float(words[6])
            deltat[il] = float(words[10])

            line = lines_P[il]
            words = line.split()
            P[il] = float(words[2])
            deltaP[il] = float(words[7])

            line = lines_X[il]
            words = line.split()
            X[il] = float(words[2])
            Y[il] = float(words[6])

        dico_results['it'] = it
        dico_results['t'] = t
        dico_results['deltat'] = deltat
        dico_results['P'] = P
        dico_results['deltaP'] = deltaP
        dico_results['X'] = X
        dico_results['Y'] = Y

        return dico_results

    def plot_deltat(self):
        dico_results = self.load()

        t = dico_results['t']
        deltat = dico_results['deltat']

        size_axe = [0.12, 0.12, 0.8, 0.8]
        fig, ax = self.output.figure_axe(size_axe=size_axe)

        ax.set_xlabel('t')
        ax.set_ylabel('deltat(t)')

        ax.set_title('info stdout, solver '+self.output.name_solver)
        ax.plot(t, deltat, 'k', linewidth=2)

    def plot_potential(self):
        dico_results = self.load()

        t = dico_results['t']
        P = dico_results['P']
        deltaP = dico_results['deltaP']
        size_axe = [0.12, 0.12, 0.8, 0.8]
        fig, ax = self.output.figure_axe(size_axe=size_axe)

        ax.set_xlabel('t')
        ax.set_ylabel('P(t), deltaP(t)')
        ax.plot(t, P, 'k', linewidth=2)
        ax.plot(t, deltaP, 'b', linewidth=2)

    def plot_XY_vs_time(self):
        dico_results = self.load()
        t = dico_results['t']
        X = dico_results['X']
        Y = dico_results['Y']

        size_axe = [0.12, 0.12, 0.8, 0.8]
        fig, ax = self.output.figure_axe(size_axe=size_axe)

        ax.set_xlabel('$t$')
        ax.set_ylabel('$X$, $Y$')

        ax.plot(t, X, 'b', label='$X$, prey')
        ax.plot(t, Y, 'r', label='$Y$, predator')

        ax.plot(ax.get_xlim(), [self.sim.Xs] * 2, 'b--')
        ax.plot(ax.get_xlim(), [self.sim.Ys] * 2, 'r--')
        ax.legend()

    def plot_XY(self):
        dico_results = self.load()
        X = dico_results['X']
        Y = dico_results['Y']

        size_axe = [0.12, 0.12, 0.8, 0.8]
        fig, ax = self.output.figure_axe(size_axe=size_axe)

        ax.set_xlabel('$X$, prey')
        ax.set_ylabel('$Y$, predator')

        ax.plot(X, Y, 'b')

        ax.plot(self.sim.Xs, self.sim.Ys, 'bx')

        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([0, ax.get_ylim()[1]])
