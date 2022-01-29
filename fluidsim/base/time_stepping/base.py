"""Time stepping (:mod:`fluidsim.base.time_stepping.base`)
================================================================

Provides:

.. autoclass:: TimeSteppingBase0
   :members:
   :private-members:

.. autoclass:: TimeSteppingBase
   :members:
   :private-members:

"""

from signal import signal
from warnings import warn
from math import pi
from datetime import datetime, timedelta
from time import time

from fluiddyn.util import mpi


def max_abs(arr):
    return max(abs(arr.min()), abs(arr.max()))


class TimeSteppingBase0:
    """Universal time stepping class used for all solvers."""

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""
        attribs = {
            "USE_T_END": True,
            "t_end": 10.0,
            "it_end": 10,
            "USE_CFL": False,
            "type_time_scheme": "RK4",
            "deltat0": 0.2,
            "deltat_max": 0.2,
            "cfl_coef": None,
            "max_elapsed": None,
        }
        params._set_child("time_stepping", attribs=attribs)

        params.time_stepping._set_doc(
            """

See :mod:`fluidsim.base.time_stepping.base`.

USE_T_END: bool (default True)

    If True, time step until t > t_end. If False, time step until it >= it_end.

t_end: float

    See documentation USE_T_END.

it_end: int

    If USE_T_END is False, number of time steps.

USE_CFL: bool (default False)

    If True, use an adaptive time step computed in particular with a
    Courant-Friedrichs-Lewy (CFL) condition.

type_time_scheme: str (default "RK4")

    Type of time scheme. Can be in ("RK2", "RK4").

deltat0: float (default 0.2)

    If USE_CFL is False, value of the time step.

deltat_max: float (default 0.2)

    Maximum value of the time step (useful when USE_CFL is True).

cfl_coef: float (default None)

    If not None, clf_coef used in the CFL condition. If None, the value is choosen
    taking into account the time scheme.

max_elapsed: number or str (default None)

    If not None, the computation stops when the elapsed time becomes larger
    than `max_elapsed`. Can be a number (in seconds) or a string (formated as
    "%H:%M:%S").

"""
        )

    def __init__(self, sim):
        self.params = sim.params
        self.sim = sim

        self.it = 0
        self.t = 0

        self._has_to_stop = False

        def handler_signals(signal_number, stack):
            print(f"signal {signal_number} received.")
            self._has_to_stop = True

        try:
            signal(12, handler_signals)
        except ValueError:
            warn("Cannot handle signals - is multithreading on?")

        try:
            param_max_elapsed = self.params.time_stepping.max_elapsed
        except AttributeError:
            # loading an old simulation?
            param_max_elapsed = None

        if param_max_elapsed is not None:
            try:
                self.max_elapsed = float(param_max_elapsed)
            except ValueError:
                t = datetime.strptime(param_max_elapsed, "%H:%M:%S")
                delta_t = timedelta(
                    hours=t.hour, minutes=t.minute, seconds=t.second
                )
                self.max_elapsed = delta_t.total_seconds()

            t_start = None
            if mpi.rank == 0:
                t_start = time()
            if mpi.nb_proc > 1:
                t_start = mpi.comm.bcast(t_start, root=0)
            self._time_should_stop = t_start + self.max_elapsed
        else:
            self.max_elapsed = None

    def start(self):
        """Loop to run the function :func:`one_time_step`.

        If ``self.USE_T_END`` is true, run till ``t >= t_end``,
        otherwise run ``self.it_end`` time steps.
        """
        self.main_loop(print_begin=True, save_init_field=True)
        self.finalize_main_loop()

    def prepare_main_loop(self):
        """Prepare the simulation just before the main loop.

        This function is called automatically in ``main_loop`` if it hasn't
        been called before. It can be used by users for debugging.

        During this preparation, the time of the begining of the simulation is
        set and the outputs are initialized with the initial state.

        """
        self.sim.__enter__()

        output = self.sim.output
        if (
            not hasattr(output, "_has_been_initialized_with_state")
            or not output._has_been_initialized_with_state
        ):
            output.init_with_initialized_state()

        self._prepare_main_loop_called = True

    def finalize_main_loop(self):
        """Finalize the simulation after the main time loop.

        - set the end time
        - finalize the outputs (in particular close the files)
        """
        self.sim.__exit__()

    def main_loop(self, print_begin=False, save_init_field=False):
        """The main time loop!"""

        if not hasattr(self, "_prepare_main_loop_called"):
            self.prepare_main_loop()

        print_stdout = self.sim.output.print_stdout
        if print_begin:
            print_stdout(
                "*************************************\n"
                "Beginning of the computation"
            )
        if save_init_field and self.sim.output._has_to_save:
            self.sim.output.phys_fields.save()

        params_stepping = self.params.time_stepping

        if params_stepping.USE_T_END:
            print_stdout(f"    compute until t = {params_stepping.t_end:10.6g}")
            while self.t < params_stepping.t_end and not self._has_to_stop:
                self.one_time_step()
        else:
            print_stdout(f"    compute until it = {params_stepping.it_end:8d}")
            while self.it < params_stepping.it_end and not self._has_to_stop:
                self.one_time_step()

    def one_time_step(self):
        """Main time stepping function."""
        if self.params.time_stepping.USE_CFL:
            self.compute_time_increment_CLF()
        if self.sim.is_forcing_enabled:
            self.sim.forcing.compute()
        if self.max_elapsed is not None:
            if mpi.rank == 0:
                now = time()
            else:
                now = None
            if mpi.nb_proc > 1:
                now = mpi.comm.bcast(now, root=0)
            if now > self._time_should_stop:
                self.sim.output.print_stdout(
                    "Maximum elapsed time reached. Should stop soon."
                )
                self._has_to_stop = True
        self.sim.output.one_time_step()
        self.one_time_step_computation()
        self.t += self.deltat
        self.it += 1


class TimeSteppingBase(TimeSteppingBase0):
    def _init_compute_time_step(self):

        params_ts = self.params.time_stepping

        if params_ts.USE_CFL:
            if params_ts.cfl_coef is not None:
                self.CFL = params_ts.cfl_coef
            elif any(
                params_ts.type_time_scheme.startswith(scheme)
                for scheme in ["RK2", "Euler"]
            ):
                self.CFL = 0.4
            elif params_ts.type_time_scheme.startswith("RK4"):
                self.CFL = 1.0
            else:
                raise ValueError("Problem name time_scheme")

        else:
            self.deltat = params_ts.deltat0

        self.deltat = params_ts.deltat0

        has_vars = self.sim.state.has_vars

        # TODO: Replace multiple function calls below when has_vars supports
        # `strict` parameter.
        has_ux = has_vars("ux") or has_vars("vx")
        has_uy = has_vars("uy") or has_vars("vy")
        has_uz = has_vars("uz") or has_vars("vz")
        has_eta = has_vars("eta")

        if has_ux and has_uy and has_uz:
            self.compute_time_increment_CLF = (
                self._compute_time_increment_CLF_uxuyuz
            )
        elif has_ux and has_uy and has_eta:
            self.compute_time_increment_CLF = (
                self._compute_time_increment_CLF_uxuyeta
            )
        elif has_ux and has_uy:
            self.compute_time_increment_CLF = (
                self._compute_time_increment_CLF_uxuy
            )
        elif has_ux:
            self.compute_time_increment_CLF = self._compute_time_increment_CLF_ux
        elif hasattr(self.params, "U"):
            self.compute_time_increment_CLF = self._compute_time_increment_CLF_U
        elif params_ts.USE_CFL:
            raise ValueError("params_ts.USE_CFL but no velocity.")

        self.deltat_max = params_ts.deltat_max

    def _init_time_scheme(self):

        params_ts = self.params.time_stepping

        if params_ts.type_time_scheme == "RK2":
            self._time_step_RK = self._time_step_RK2
        elif params_ts.type_time_scheme == "RK4":
            self._time_step_RK = self._time_step_RK4
        else:
            raise ValueError("Problem name time_scheme")

    def is_simul_completed(self):
        """Checks if simulation time or iteration has reached the end according
        to parameters specified.

        """
        if self.params.time_stepping.USE_T_END:
            return self.t >= self.params.time_stepping.t_end

        else:
            return self.it >= self.params.time_stepping.it_end

    def _compute_time_increment_CLF_uxuyuz(self):
        """Compute the time increment deltat with a CLF condition."""
        get_var = self.sim.state.get_var
        ux = get_var("vx")
        uy = get_var("vy")
        uz = get_var("vz")

        if ux.size > 0:
            max_ux = max_abs(ux)
            max_uy = max_abs(uy)
            max_uz = max_abs(uz)
            tmp = (
                max_ux / self.sim.oper.deltax
                + max_uy / self.sim.oper.deltay
                + max_uz / self.sim.oper.deltaz
            )
        else:
            tmp = 0.0

        self._compute_time_increment_CLF_from_tmp(tmp)

    def _compute_time_increment_CLF_from_tmp(self, tmp):

        if mpi.nb_proc > 1:
            tmp = mpi.comm.allreduce(tmp, op=mpi.MPI.MAX)

        if tmp > 0:
            deltat_CFL = self.CFL / tmp
        else:
            deltat_CFL = self.deltat_max

        maybe_new_dt = min(deltat_CFL, self.deltat_max)
        normalize_diff = abs(self.deltat - maybe_new_dt) / maybe_new_dt

        if normalize_diff > 0.02:
            self.deltat = maybe_new_dt

    def _compute_time_increment_CLF_uxuy(self):
        """Compute the time increment deltat with a CLF condition."""

        ux = self.sim.state.get_var("ux")
        uy = self.sim.state.get_var("uy")

        max_ux = max_abs(ux)
        max_uy = max_abs(uy)
        tmp = max_ux / self.sim.oper.deltax + max_uy / self.sim.oper.deltay

        self._compute_time_increment_CLF_from_tmp(tmp)

    def _compute_time_increment_CLF_uxuyeta(self):
        """Compute the time increment deltat with a CLF condition."""

        ux = self.sim.state.get_var("ux")
        uy = self.sim.state.get_var("uy")

        params = self.sim.params
        try:
            f = params.f
        except AttributeError:
            # For spherical solvers, trying to use the dispersion relation for
            # Poincare waves can give absurd phase speeds due to earth radius
            # coming in the relation for wavenumbers kh_l = l * (l + 1) / r
            # f = params.omega
            f = 0

        # Phase speed of the fastest wave from dispersion relation
        if f == 0:
            cph = params.c2**0.5
        else:
            Lh = max(params.oper.Lx, params.oper.Ly)
            k_min = 2 * pi / Lh

            cph = (f**2 / k_min**2 + params.c2) ** 0.5

        max_ux = max_abs(ux)
        max_uy = max_abs(uy)
        tmp = max_ux / self.sim.oper.deltax + max_uy / self.sim.oper.deltay

        if mpi.nb_proc > 1:
            tmp = mpi.comm.allreduce(tmp, op=mpi.MPI.MAX)

        if tmp > 0:
            deltat_CFL = self.CFL / tmp
        else:
            deltat_CFL = self.deltat_max

        deltat_wave = (
            self.CFL * min(self.sim.oper.deltax, self.sim.oper.deltay) / cph
        )
        maybe_new_dt = min(deltat_CFL, deltat_wave, self.deltat_max)
        normalize_diff = abs(self.deltat - maybe_new_dt) / maybe_new_dt

        if normalize_diff > 0.02:
            self.deltat = maybe_new_dt

    def _compute_time_increment_CLF_ux(self):
        """Compute the time increment deltat with a CLF condition."""
        ux = self.sim.state.get_var("ux")
        max_ux = max_abs(ux)
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
