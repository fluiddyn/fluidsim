"""Time stepping (:mod:`fluidsim.base.basilisk.time_stepping`)
==============================================================


Provides:

.. autoclass:: TimeSteppingBasilisk
   :members:
   :private-members:

"""

from time import time
from signal import signal

import numpy as np


class TimeSteppingBasilisk:
    """Time stepping class to handle Basilisk's event loop and FluidSim output."""

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""
        attribs = {"USE_T_END": True, "t_end": 10.0, "it_end": 10, "deltat0": 0.5}
        params._set_child("time_stepping", attribs=attribs)

    def __init__(self, sim):
        self.params = sim.params
        self.sim = sim

        self.it = 0
        self.t = 0

        self._has_to_stop = False

        def handler_signals(signal_number, stack):
            print(f"signal {signal_number} received.")
            self._has_to_stop = True

        signal(12, handler_signals)

    def start(self):
        output = self.sim.output
        output.init_with_initialized_state()

        print_stdout = output.print_stdout
        print_stdout(
            "*************************************\n"
            + "Beginning of the computation"
        )
        if output._has_to_save:
            output.phys_fields.save()

        params_ts = self.sim.params.time_stepping
        if params_ts.USE_T_END:
            t_end = params_ts.t_end
            nb_time_steps = int(float(params_ts.t_end) / params_ts.deltat0)
            self.deltat = t_end / nb_time_steps
        else:
            nb_time_steps = params_ts.it_end
            t_end = params_ts.deltat0 * params_ts.it_end
            self.deltat = params_ts.deltat0

        def one_time_step(i, t):
            print(f"Basilisk one_time_step: (i, t) = {i}, {t}")
            self.t = t
            self.it = i
            self.sim.state._get_state_from_basilisk()
            self.sim.output.one_time_step()

        print("t_end, nb_time_steps", t_end, nb_time_steps)

        self.sim.basilisk.event(
            one_time_step, t=np.linspace(0.0, t_end, nb_time_steps + 1)
        )

        time_begining_simul = time()

        self.sim.basilisk.run()

        total_time_simul = time() - time_begining_simul
        output.end_of_simul(total_time_simul)
