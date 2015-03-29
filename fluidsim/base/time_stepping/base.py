"""Time stepping (:mod:`fluidsim.base.time_stepping.base`)
================================================================

.. currentmodule:: fluidsim.base.time_stepping.base

Provides:

.. autoclass:: TimeSteppingBase
   :members:
   :private-members:

"""

from time import time

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
                   'USE_CFL': True,
                   'type_time_scheme': 'RK4',
                   'deltat0': 0.2}
        params.set_child('time_stepping', attribs=attribs)

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

        has_ux = self.sim.state.can_this_key_be_obtained('ux')
        has_uy = self.sim.state.can_this_key_be_obtained('uy')
        has_uz = self.sim.state.can_this_key_be_obtained('uz')

        if has_ux and has_uy and has_uz:
            self._compute_time_increment_CLF = \
                self._compute_time_increment_CLF_uxuyuz
        elif has_ux and has_uy:
            self._compute_time_increment_CLF = \
                self._compute_time_increment_CLF_uxuy
        elif has_ux:
            self._compute_time_increment_CLF = \
                self._compute_time_increment_CLF_ux
        else:
            self._compute_time_increment_CLF = \
                self._compute_time_increment_CLF_no_ux

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
        print_stdout = self.sim.output.print_stdout
        print_stdout(
            '*************************************\n' +
            'Beginning of the computation')
        if self.sim.output.has_to_save:
            self.sim.output.phys_fields.save()
        time_begining_simul = time()
        if self.params.time_stepping.USE_T_END:
            print_stdout(
                '    compute until t = {0:10.6g}'.format(
                    self.params.time_stepping.t_end))
            while self.t < self.params.time_stepping.t_end:
                self.one_time_step()
        else:
            print_stdout(
                '    compute until it = {0:8d}'.format(
                    self.params.time_stepping.it_end))
            while self.it < self.params.time_stepping.it_end:
                self.one_time_step()
        total_time_simul = time() - time_begining_simul
        self.sim.output.end_of_simul(total_time_simul)

    def one_time_step(self):
        if self.params.time_stepping.USE_CFL:
            self._compute_time_increment_CLF()
        if self.params.FORCING:
            self.sim.forcing.compute()
        self.sim.output.one_time_step()
        self.one_time_step_computation()

    def _compute_time_increment_CLF_uxuyuz(self):
        """Compute the time increment deltat with a CLF condition."""

        ux = self.sim.state('ux')
        uy = self.sim.state('uy')
        uz = self.sim.state('uz')

        max_ux = abs(ux).max()
        max_uy = abs(uy).max()
        max_uz = abs(uz).max()
        temp = (max_ux/self.sim.oper.deltax +
                max_uy/self.sim.oper.deltay +
                max_uz/self.sim.oper.deltaz)

        if mpi.nb_proc > 1:
            temp = mpi.comm.allreduce(temp, op=mpi.MPI.MAX)

        if temp > 0:
            deltat_CFL = self.CFL/temp
        else:
            deltat_CFL = self.deltat_max

        maybe_new_dt = min(deltat_CFL, self.deltat_max)
        normalize_diff = abs(self.deltat-maybe_new_dt)/maybe_new_dt

        if normalize_diff > 0.02:
            self.deltat = maybe_new_dt

    def _compute_time_increment_CLF_uxuy(self):
        """Compute the time increment deltat with a CLF condition."""

        ux = self.sim.state('ux')
        uy = self.sim.state('uy')

        max_ux = abs(ux).max()
        max_uy = abs(uy).max()
        temp = (max_ux/self.sim.oper.deltax + max_uy/self.sim.oper.deltay)

        if mpi.nb_proc > 1:
            temp = mpi.comm.allreduce(temp, op=mpi.MPI.MAX)

        if temp > 0:
            deltat_CFL = self.CFL/temp
        else:
            deltat_CFL = self.deltat_max

        maybe_new_dt = min(deltat_CFL, self.deltat_max)
        normalize_diff = abs(self.deltat-maybe_new_dt)/maybe_new_dt

        if normalize_diff > 0.02:
            self.deltat = maybe_new_dt

    def _compute_time_increment_CLF_ux(self):
        """Compute the time increment deltat with a CLF condition."""
        ux = self.sim.state('ux')
        max_ux = abs(ux).max()
        temp = max_ux/self.sim.oper.deltax

        if mpi.nb_proc > 1:
            temp = mpi.comm.allreduce(temp, op=mpi.MPI.MAX)

        if temp > 0:
            deltat_CFL = self.CFL/temp
        else:
            deltat_CFL = self.deltat_max

        maybe_new_dt = min(deltat_CFL, self.deltat_max)
        normalize_diff = abs(self.deltat-maybe_new_dt)/maybe_new_dt

        if normalize_diff > 0.02:
            self.deltat = maybe_new_dt

    def _compute_time_increment_CLF_no_ux(self):
        """Compute the time increment deltat with a CLF condition."""
        max_ux = self.params.U
        temp = max_ux/self.sim.oper.deltax

        if mpi.nb_proc > 1:
            temp = mpi.comm.allreduce(temp, op=mpi.MPI.MAX)

        if temp > 0:
            deltat_CFL = self.CFL/temp
        else:
            deltat_CFL = self.deltat_max

        maybe_new_dt = min(deltat_CFL, self.deltat_max)
        normalize_diff = abs(self.deltat-maybe_new_dt)/maybe_new_dt

        if normalize_diff > 0.02:
            self.deltat = maybe_new_dt
