"""Time stepping (:mod:`fluidsim.base.time_stepping.base`)
================================================================

Provides:

.. autoclass:: TimeSteppingBase
   :members:
   :private-members:

"""
from __future__ import division
from __future__ import print_function

from builtins import object

from signal import signal
from math import pi

from fluiddyn.util import mpi


class TimeSteppingBase(object):
    """Universal time stepping class used for all solvers.


    """
    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        attribs = {'USE_T_END': True,
                   't_end': 10.,
                   'it_end': 10,
                   'USE_CFL': False,
                   'type_time_scheme': 'RK4',
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

    def _init_compute_time_step(self):

        params_ts = self.params.time_stepping

        if params_ts.USE_CFL:
            if params_ts.type_time_scheme == 'RK2':
                self.CFL = 0.4
            elif params_ts.type_time_scheme == 'RK4':
                self.CFL = 1.0
            else:
                raise ValueError('Problem name time_scheme')
        else:
            self.deltat = params_ts.deltat0

        self.deltat = params_ts.deltat0

        has_vars = self.sim.state.has_vars

        # TODO: Replace multiple function calls below when has_vars supports
        # `strict` parameter.
        has_ux = has_vars('ux') or has_vars('vx')
        has_uy = has_vars('uy') or has_vars('vy')
        has_uz = has_vars('uz') or has_vars('vz')
        has_eta = has_vars('eta')

        if has_ux and has_uy and has_uz:
            self._compute_time_increment_CLF = \
                self._compute_time_increment_CLF_uxuyuz
        elif has_ux and has_uy and has_eta:
            self._compute_time_increment_CLF = \
                self._compute_time_increment_CLF_uxuyeta
        elif has_ux and has_uy:
            self._compute_time_increment_CLF = \
                self._compute_time_increment_CLF_uxuy
        elif has_ux:
            self._compute_time_increment_CLF = \
                self._compute_time_increment_CLF_ux
        elif hasattr(self.params, 'U'):
            self._compute_time_increment_CLF = \
                self._compute_time_increment_CLF_U
        elif params_ts.USE_CFL:
            raise ValueError('params_ts.USE_CFL but no velocity.')

        self.deltat_max = 0.2

    def _init_time_scheme(self):

        params_ts = self.params.time_stepping

        if params_ts.type_time_scheme == 'RK2':
            self._time_step_RK = self._time_step_RK2
        elif params_ts.type_time_scheme == 'RK4':
            self._time_step_RK = self._time_step_RK4
        else:
            raise ValueError('Problem name time_scheme')

    def start(self):
        """Loop to run the function :func:`one_time_step`.

        If *self.USE_T_END* is true, run till ``t >= t_end``,
        otherwise run *self.it_end* time steps.
        """
        self.sim.__enter__()

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
        if self.params.time_stepping.USE_T_END:
            print_stdout(
                '    compute until t = {0:10.6g}'.format(
                    self.params.time_stepping.t_end))
            while (self.t < self.params.time_stepping.t_end and
                   not self._has_to_stop):
                self.one_time_step()
        else:
            print_stdout(
                '    compute until it = {0:8d}'.format(
                    self.params.time_stepping.it_end))
            while (self.it < self.params.time_stepping.it_end and
                   not self._has_to_stop):
                self.one_time_step()

        self.sim.__exit__()

    def is_simul_completed(self):
        """Checks if simulation time or iteration has reached the end according
        to parameters specified.

        """
        if self.params.time_stepping.USE_T_END:
            return self.t >= self.params.time_stepping.t_end
        else:
            return self.it >= self.params.time_stepping.it_end

    def one_time_step(self):
        """Main time stepping function."""
        if self.params.time_stepping.USE_CFL:
            self._compute_time_increment_CLF()
        if self.sim.is_forcing_enabled:
            self.sim.forcing.compute()
        self.sim.output.one_time_step()
        self.one_time_step_computation()
        self.t += self.deltat
        self.it += 1

    def _compute_time_increment_CLF_uxuyuz(self):
        """Compute the time increment deltat with a CLF condition."""
        get_var = self.sim.state.get_var

        ux = get_var('vx')
        uy = get_var('vy')
        uz = get_var('vz')

        max_ux = abs(ux).max()
        max_uy = abs(uy).max()
        max_uz = abs(uz).max()
        tmp = (max_ux / self.sim.oper.deltax +
               max_uy / self.sim.oper.deltay +
               max_uz / self.sim.oper.deltaz)

        self._compute_time_increment_CLF_from_tmp(tmp)

    def _compute_time_increment_CLF_from_tmp(self, tmp):

        if mpi.nb_proc > 1:
            tmp = mpi.comm.allreduce(tmp, op=mpi.MPI.MAX)

        if tmp > 0:
            deltat_CFL = self.CFL / tmp
        else:
            deltat_CFL = self.deltat_max

        maybe_new_dt = min(deltat_CFL, self.deltat_max)
        normalize_diff = abs(self.deltat-maybe_new_dt) / maybe_new_dt

        if normalize_diff > 0.02:
            self.deltat = maybe_new_dt

    def _compute_time_increment_CLF_uxuy(self):
        """Compute the time increment deltat with a CLF condition."""

        ux = self.sim.state.get_var('ux')
        uy = self.sim.state.get_var('uy')

        max_ux = abs(ux).max()
        max_uy = abs(uy).max()
        tmp = (max_ux / self.sim.oper.deltax + max_uy / self.sim.oper.deltay)

        self._compute_time_increment_CLF_from_tmp(tmp)

    def _compute_time_increment_CLF_uxuyeta(self):
        """Compute the time increment deltat with a CLF condition."""

        ux = self.sim.state.get_var('ux')
        uy = self.sim.state.get_var('uy')

        params = self.sim.params
        f = params.f
        c = params.c2 ** 0.5
        if f != 0:
            Lh = max(params.oper.Lx, params.oper.Ly)
            k_min = 2 * pi / Lh
            c = (f ** 2 / k_min ** 2 + c ** 2) ** 0.5

        max_ux = abs(ux).max()
        max_uy = abs(uy).max()
        tmp = max_ux / self.sim.oper.deltax + max_uy / self.sim.oper.deltay

        if mpi.nb_proc > 1:
            tmp = mpi.comm.allreduce(tmp, op=mpi.MPI.MAX)

        if tmp > 0:
            deltat_CFL = self.CFL / tmp
        else:
            deltat_CFL = self.deltat_max

        deltat_wave = self.CFL * min(self.sim.oper.deltax, self.sim.oper.deltay) / c
        maybe_new_dt = min(deltat_CFL, deltat_wave, self.deltat_max)
        normalize_diff = abs(self.deltat-maybe_new_dt)/maybe_new_dt

        if normalize_diff > 0.02:
            self.deltat = maybe_new_dt

    def _compute_time_increment_CLF_ux(self):
        """Compute the time increment deltat with a CLF condition."""
        ux = self.sim.state.get_var('ux')
        max_ux = abs(ux).max()
        tmp = max_ux / self.sim.oper.deltax
        self._compute_time_increment_CLF_from_tmp(tmp)

    def _compute_time_increment_CLF_U(self):
        """Compute the time increment deltat with a CLF condition."""
        max_ux = self.params.U
        tmp = max_ux / self.sim.oper.deltax
        self._compute_time_increment_CLF_from_tmp(tmp)

    def _compute_dispersion_relation(self):
        """Compute time increment from a dispersion relation."""
        pass
