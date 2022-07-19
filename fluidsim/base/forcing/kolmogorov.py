"""Kolmogorov flow (:mod:`fluidsim.base.forcing.kolmogorov`)
============================================================

.. autoclass:: KolmogorovFlow
   :members:
   :private-members:

.. autoclass:: KolmogorovFlowNormalized
   :members:
   :private-members:

"""
from warnings import warn

import numpy as np

from fluiddyn.util import mpi

from fluidsim_core.extend_simul import SimulExtender, extend_simul_class

from fluidsim.base.forcing.specific import (
    SpecificForcingPseudoSpectralSimple,
    NormalizedForcing,
)

__all__ = ["extend_simul_class", "KolmogorovFlow", "KolmogorovFlowNormalized"]


class _KolmogorovFlowBase(SimulExtender):
    _module_name = "fluidsim.base.forcing.kolmogorov"
    tag = "kolmogorov_flow"

    @classmethod
    def get_modif_info_solver(cls):
        def modif_info_solver(info_solver):
            info_solver.classes.Forcing.classes._set_child(
                cls.tag,
                attribs={
                    "module_name": cls._module_name,
                    "class_name": cls.__name__,
                },
            )

        return modif_info_solver

    @classmethod
    def complete_params_with_default(cls, params):
        params.forcing.available_types.append(cls.tag)
        if not hasattr(params.forcing, "kolmo"):
            params.forcing._set_child(
                "kolmo",
                attribs={"ik": 3, "amplitude": None, "letter_gradient": None},
            )

    def __init__(self, sim):

        if len(sim.oper.axes) == 3:
            self._key_forced_default = "vx_fft"
        else:
            self._key_forced_default = "ux_fft"

        super().__init__(sim)

        if self.tag == "kolmogorov_flow_normalized" and mpi.rank > 0:
            return

        params = sim.params

        ik = params.forcing.kolmo.ik
        amplitude = params.forcing.kolmo.amplitude

        if self.tag == "kolmogorov_flow_normalized" and amplitude is not None:
            warn(
                "For this forcing type ('kolmogorov_flow_normalized'), "
                "the parameter `params.forcing.kolmo.amplitude` has no effect. "
                "The amplitude is controlled with `params.forcing.forcing_rate`."
            )

        if amplitude is None:
            amplitude = 1.0

        key_forced = params.forcing.key_forced
        letter_gradient = params.forcing.kolmo.letter_gradient

        oper = self._get_oper()

        if len(oper.axes) == 3:
            coords = oper.get_XYZ_loc()
            lengths = [params.oper.Lx, params.oper.Ly, params.oper.Lz]
            letters = "xyz"
            if key_forced is None:
                key_forced = "vx"
        else:
            coords = [oper.X, oper.Y]
            lengths = [params.oper.Lx, params.oper.Ly]
            letters = "xy"
            if key_forced is None:
                key_forced = "ux"

        if key_forced.endswith("_fft"):
            key_forced = key_forced[: -len("_fft")]

        if letter_gradient is None:
            letter_gradient = letters[-1]

        if letter_gradient not in letters:
            raise ValueError

        index = letters.index(letter_gradient)
        variable = coords[index]
        length = lengths[index]

        self._init_forcing(key_forced, amplitude, ik, length, variable)

    def _get_oper(self):
        raise NotImplementedError

    def _init_forcing(self, key_forced, amplitude, ik, length, variable):
        raise NotImplementedError


class KolmogorovFlow(_KolmogorovFlowBase, SpecificForcingPseudoSpectralSimple):
    """Kolmogorov flow forcing

    This forcing can be used with solvers based on ns2d or ns3d.

    By default, the flow is forced along the x axis with a gradient along the z
    axis for 3D solvers and the y axis for the 2D solvers.

    Examples
    --------

    .. code-block:: python

        from fluidsim.solvers.ns3d.solver import Simul as SimulNotExtended

        from fluidsim.base.forcing.kolmogorov import extend_simul_class, KolmogorovFlow

        Simul = extend_simul_class(SimulNotExtended, KolmogorovFlow)

        params = Simul.create_default_params()

        params.forcing.enable = True
        params.forcing.type = "kolmogorov_flow"

    The parameters can be modified as follow:

    .. code-block:: python

        params.forcing.kolmo.ik = 3
        params.forcing.kolmo.amplitude = 2.0

        params.forcing.key_forced = "ux_fft"
        params.forcing.kolmo.letter_gradient = "y"

    """

    def _get_oper(self):
        return self.sim.oper

    def _init_forcing(self, key_forced, amplitude, ik, length, variable):
        self.fstate.init_statephys_from(
            **{key_forced: amplitude * np.sin(2 * np.pi * ik / length * variable)}
        )
        self.fstate.statespect_from_statephys()

    def compute(self):
        # nothing to do here
        pass


class KolmogorovFlowNormalized(_KolmogorovFlowBase, NormalizedForcing):
    """Kolmogorov flow forcing with constant energy injection rate

    This forcing can be used with solvers based on ns2d or ns3d.

    By default, the flow is forced along the x axis with a gradient along the z
    axis for 3D solvers and the y axis for the 2D solvers.

    Examples
    --------

    .. code-block:: python

        from fluidsim.solvers.ns3d.solver import Simul as SimulNotExtended

        from fluidsim.base.forcing.kolmogorov import extend_simul_class, KolmogorovFlowNormalized

        Simul = extend_simul_class(SimulNotExtended, KolmogorovFlowNormalized)

        params = Simul.create_default_params()

        params.forcing.enable = True
        params.forcing.type = "kolmogorov_flow_normalized"

    The parameters can be modified as follow:

    .. code-block:: python

        params.forcing.kolmo.ik = 3
        params.forcing.forcing_rate = 10.0

    Note that ``params.forcing.kolmo.amplitude`` has no effect for this forcing
    type.

    """

    tag = "kolmogorov_flow_normalized"

    def _get_oper(self):
        return self.oper_coarse

    def _init_forcing(self, key_forced, amplitude, ik, length, variable):
        del key_forced
        field = amplitude * np.sin(2 * np.pi * ik / length * variable)
        self._f_fft = self.oper_coarse.fft(field)

    def forcingc_raw_each_time(self, _):
        # nothing to do here
        return self._f_fft
