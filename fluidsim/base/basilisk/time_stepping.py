
from time import time

from signal import signal


class TimeSteppingBasilisk(object):

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        attribs = {'USE_T_END': True,
                   't_end': 10.,
                   'it_end': 10,
                   'deltat0': 0.2}
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
        if (not hasattr(output, 'has_been_initialized_with_state') or
                not output.has_been_initialized_with_state):
            output.init_with_initialized_state()

        print_stdout = output.print_stdout
        print_stdout(
            '*************************************\n' +
            'Beginning of the computation')
        if self.sim.output.has_to_save:
            self.sim.output.phys_fields.save()
        time_begining_simul = time()
        # if self.params.time_stepping.USE_T_END:
        #     print_stdout(
        #         '    compute until t = {0:10.6g}'.format(
        #             self.params.time_stepping.t_end))
        #     while (self.t < self.params.time_stepping.t_end and
        #            not self._has_to_stop):
        #         self.one_time_step()
        # else:
        #     print_stdout(
        #         '    compute until it = {0:8d}'.format(
        #             self.params.time_stepping.it_end))
        #     while (self.it < self.params.time_stepping.it_end and
        #            not self._has_to_stop):
        #         self.one_time_step()
        total_time_simul = time() - time_begining_simul
        self.sim.output.end_of_simul(total_time_simul)
