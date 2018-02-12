
from __future__ import print_function

from time import time
from signal import signal

import numpy as np


class TimeSteppingBasilisk(object):

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        attribs = {'USE_T_END': True,
                   't_end': 10.,
                   'it_end': 10,
                   'deltat0': 0.5}
        params._set_child('time_stepping', attribs=attribs)

    def __init__(self, sim):
        self.params = sim.params
        self.sim = sim

        self.it = 0
        self.t = 0

        self._has_to_stop = False

        def handler_signals(signal_number, stack):
            print('signal {} received.'.format(signal_number))
            self._has_to_stop = True

        signal(12, handler_signals)

    def start(self):
        output = self.sim.output
        if (not hasattr(output, '_has_been_initialized_with_state') or
                not output._has_been_initialized_with_state):
            output.init_with_initialized_state()

        print_stdout = output.print_stdout
        print_stdout(
            '*************************************\n' +
            'Beginning of the computation')
        if self.sim.output._has_to_save:
            self.sim.output.phys_fields.save()

        pts = self.sim.params.time_stepping
        if pts.USE_T_END:
            t_end = pts.t_end
            nb_time_steps = int(float(pts.t_end)/pts.deltat0)
            self.deltat = t_end / nb_time_steps
        else:
            nb_time_steps = pts.it_end
            t_end = pts.deltat0 * pts.it_end
            self.deltat = pts.deltat0

        def one_time_step(i, t):
            print('Basilisk one_time_step: (i, t) = '
                  '{}, {}'.format(i, t))
            self.t = t
            self.it = i
            self.sim.state._get_state_from_basilisk()
            self.sim.output.one_time_step()

        print('t_end, nb_time_steps', t_end, nb_time_steps)

        self.sim.basilisk.event(
            one_time_step, t=np.linspace(0., t_end, nb_time_steps + 1))

        time_begining_simul = time()

        self.sim.basilisk.run()

        total_time_simul = time() - time_begining_simul
        self.sim.output.end_of_simul(total_time_simul)
