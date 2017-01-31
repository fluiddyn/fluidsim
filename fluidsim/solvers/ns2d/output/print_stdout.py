"""Simple text output (:mod:`fluidsim.solvers.ns2d.output.print_stdout`)
========================================================================

.. autoclass:: PrintStdOutNS2D
   :members:
   :private-members:

"""

from __future__ import print_function, division

from builtins import range
import numpy as np

from fluidsim.base.output.print_stdout import PrintStdOutBase

from fluiddyn.util import mpi


class PrintStdOutNS2D(PrintStdOutBase):
    """Simple text output.

    Used to print in both the stdout and the stdout.txt file, and also
    to print simple info on the current state of the simulation.

    """

    def _make_str_info(self):
        to_print = super(PrintStdOutNS2D, self)._make_str_info()

        energy = self.output.compute_energy()

        if hasattr(self, 'energy_tmp'):
            delta_energy = energy-self.energy_tmp
        else:
            delta_energy = 0.

        if mpi.rank == 0:
            to_print += (
                '              energy = {:9.3e} ; Delta energy = {:+9.3e}'
                ''.format(energy, delta_energy))

            duration_left = self._evaluate_duration_left()
            if duration_left is not None:
                to_print += (
                    '\n              estimated remaining duration = {:9.3g} s'
                    ''.format(duration_left))

        self.energy_tmp = energy
        return to_print

    def load(self):
        dico_results = {'name_solver': self.output.name_solver}
        with open(self.output.path_run + '/stdout.txt') as file_means:
            lines = file_means.readlines()

        lines_t = []
        lines_E = []
        for il, line in enumerate(lines):
            if line[0:4] == 'it =':
                lines_t.append(line)
            if line[0:22] == '              energy =':
                lines_E.append(line)

        nt = len(lines_t)
        if nt > 1:
            nt -= 1

        it = np.zeros(nt, dtype=np.int)
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

        dico_results['it'] = it
        dico_results['t'] = t
        dico_results['deltat'] = deltat
        dico_results['E'] = E
        dico_results['deltaE'] = deltaE

        return dico_results

    def plot_deltat(self):
        dico_results = self.load()

        t = dico_results['t']
        deltat = dico_results['deltat']

        fig, ax = self.output.figure_axe()
        ax.set_xlabel('t')
        ax.set_ylabel('deltat(t)')

        ax.set_title('info stdout, solver ' + self.output.name_solver +
                     ', nh = {0:5d}'.format(self.sim.oper.nx_seq))
        ax.plot(t, deltat, 'k', linewidth=2)
        fig.tight_layout()

    def plot_energy(self):
        dico_results = self.load()

        t = dico_results['t']
        E = dico_results['E']
        deltaE = dico_results['deltaE']

        fig, ax = self.output.figure_axe()

        ax.set_title('info stdout, solver ' + self.output.name_solver +
                     ', nh = {0:5d}'.format(self.sim.oper.nx_seq))

        ax.set_xlabel('t')
        ax.set_ylabel('E(t), deltaE(t)')
        ax.plot(t, E, 'k', linewidth=2)
        ax.plot(t, deltaE, 'b', linewidth=2)
        fig.tight_layout()
